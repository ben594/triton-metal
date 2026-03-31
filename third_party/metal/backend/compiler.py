import os
import subprocess
import tempfile

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.metal.msl_generator import MSLKernel


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
