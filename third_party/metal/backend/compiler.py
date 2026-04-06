import functools
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from triton import knobs
from triton._C.libtriton import ir, llvm, metal, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.metal.msl_generator import MSLKernel


def get_min_dot_size(target: GPUTarget):
    # TODO copied from AMD, modify if needed
    return lambda lhs_type, rhs_type: (1, 1, 1)


@dataclass(frozen=True)
class MetalOptions:
    # TODO add more metal-specific options as needed
    backend_name: str = "metal"
    num_warps: int = 4
    num_ctas: int = 1
    warp_size: int = 32  # SIMD group size
    sanitize_overflow: bool = True  # TODO copied from AMD, modify if needed
    debug: bool = False
    extern_libs: Optional[dict] = None
    arch: Optional[str] = None
    instrumentation_mode: str = ""
    enable_fp_fusion: bool = True

    supported_fp8_dtypes: Tuple = ()  # TODO I believe mac does not support fp8
    launch_cooperative_grid: bool = False

    def __post_init__(self):
        # TODO verify this
        default_libdir = Path(__file__).parent / "lib"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))

    def hash(self):
        # TODO copied from AMD, modify if needed
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "metal"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # TODO copied from triton-cpu/python/triton/backends/cpu/compiler.py, need to verify
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # TODO what to put for architecture label
        passes.ttir.add_convert_to_ttgpuir(pm, "metal", options.num_warps, options.warp_size, options.num_ctas)
        pm.run(mod, "make_ttgir_early")
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # TODO insert passes
        pm.run(mod, "make_ttgir")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_air(src, metadata, options):
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        metal.passes.ttgpuir.add_to_llvmir(pm, str(options.arch))
        passes.common.add_canonicalizer(pm)

        pm.run(mod, "make_llir")

        print("Finished making llir")
        print("MOD\n", mod)

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        # print("MOD\n", mod)
        llvm_mod = llvm.to_module(mod, context)
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_msl(src, metadata, opt) -> str:
        mod = src
        # TODO check what is in metadata and opt
        metadata["shared"] = 0
        metadata["global_scratch_size"] = 0
        metadata["global_scratch_align"] = 1
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1
        msl_kernel = MSLKernel(mod)
        msl_code: str = msl_kernel.generate_msl_code()
        metadata["name"] = msl_kernel.name
        metadata["num_warps"] = 4  # TODO just took 4 from an example, what should this be?
        return msl_code

    @staticmethod
    def make_metallib(src: str, metadata, opt) -> bytes:
        # TODO check what is in metadata and opt
        with tempfile.TemporaryDirectory() as tmpdir:
            air_path = os.path.join(tmpdir, "kernel.air")
            lib_path = os.path.join(tmpdir, "kernel.metallib")

            with open(air_path, "w") as f:
                f.write(src)

            # .air -> .metallib
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", lib_path], check=True)
            with open(lib_path, "rb") as f:
                return f.read()

    def add_stages(self, stages: dict, options: object, language: Language) -> None:
        assert language == Language.TRITON
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["air"] = lambda src, metadata: self.make_air(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO modify if needed, currently just return the target
        return f"{self.target}"

    def parse_options(self, opts: dict) -> MetalOptions:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        args = {"arch": knobs.runtime.override_arch or self.target.arch}
        args.update({k: opts[k] for k in MetalOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})

        if args.get("num_ctas", 1) > 1 and not metal.supports_multi_cta_launch(self.target.arch):
            raise ValueError(f"num_ctas > 1 not supported on {self.target.arch}")

        if "supported_fp8_dtypes" not in args:
            args["supported_fp8_dtypes"] = tuple(sorted(MetalOptions.supported_fp8_dtypes))

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        # TODO modify if needed
        return metadata

    def get_codegen_implementation(self, options):
        # TODO copied from AMD backend, modify if needed
        return {"min_dot_size": get_min_dot_size(self.target)}

    def load_dialects(self, context):
        # TODO no dialects for now, add if needed
        pass

    def get_module_map(self) -> dict:
        # TODO no additional modules for now, add if needed
        return {}
