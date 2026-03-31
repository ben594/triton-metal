from dataclasses import dataclass
import functools
import os
import subprocess
import tempfile

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.metal.msl_generator import MSLKernel


def get_min_dot_size(target: GPUTarget):
    # TODO copied from AMD, modify if needed
    return lambda lhs_type, rhs_type: (1, 1, 1)


@dataclass(frozen=True)
class MetalOptions:
    # TODO add options for metal as needed
    pass


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
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_msl(src, metadata, opt) -> str:
        mod = src
        # TODO check what is in metadata and opt
        msl_kernel = MSLKernel(mod)
        msl_code: str = msl_kernel.generate_msl_code()
        return msl_code

    @staticmethod
    def make_metallib(src: str, metadata, opt) -> bytes:
        # TODO check what is in metadata and opt
        with tempfile.TemporaryDirectory() as tmpdir:
            msl_path = os.path.join(tmpdir, "kernel.metal")
            air_path = os.path.join(tmpdir, "kernel.air")
            lib_path = os.path.join(tmpdir, "kernel.metallib")

            with open(msl_path, "w") as f:
                f.write(src)

            # .metal -> .air
            subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", msl_path, "-o", air_path], check=True)
            # .air -> .metallib
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", lib_path], check=True)
            with open(lib_path, "rb") as f:
                return f.read()

    def add_stages(self, stages: dict, options: object, language: Language) -> None:
        assert language == Language.TRITON
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        # TODO modify if needed, currently just return the target
        return f"{self.target}"

    def parse_options(self, options: dict) -> MetalOptions:
        args = {k: options[k] for k in MetalOptions.__dataclass_fields__.keys() if k in options and options[k] is not None}
        unsupported = sorted(k for k, v in options.items() if k not in MetalOptions.__dataclass_fields__ and v is not None)
        if unsupported:
            raise ValueError(f"Unsupported Metal backend option(s): {', '.join(unsupported)}")
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
