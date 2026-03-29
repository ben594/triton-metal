from triton._C.libtriton import cpu, ir, llvm, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language


class MetalBackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "metal"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)

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
    def make_ttcir(mod, metadata, opt):
        # TODO copied from triton-cpu/python/triton/backends/cpu/compiler.py, need to verify
        # TTIR -> TTCIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttcpuir.add_scalarize(pm, True)
        cpu.passes.ttcpuir.add_convert_memory_ops(pm, True)
        cpu.passes.ttcpuir.add_convert_ptr_ops(pm)
        cpu.passes.ttcpuir.add_convert_elementwise_ops(pm)
        cpu.passes.ttcpuir.add_convert_elem_manip_ops(pm)
        cpu.passes.ttcpuir.add_convert_dot_op(pm)
        cpu.passes.ttcpuir.add_convert_histogram_op(pm)
        cpu.passes.ttcpuir.add_convert_reduction_op(pm, True, False)
        cpu.passes.ttcpuir.add_convert_scan_op(pm)
        cpu.passes.ttcpuir.add_convert_cf_ops(pm)
        cpu.passes.ttcpuir.add_convert_atomic_ops(pm)
        cpu.passes.ttcpuir.add_convert_debug_ops(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, "make_ttcir")
        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])
        return mod

    def make_llir(self, src, metadata, options):
        raise NotImplementedError("Make LLIR is not implemented for Metal backend yet")

    def make_dxil(self, src, metadata, options):
        raise NotImplementedError("Make DXIL is not implemented for Metal backend yet")

    def make_metallib(self, src, metadata, options):
        raise NotImplementedError("Make metallib is not implemented for Metal backend yet")

    def add_stages(self, stages: dict, options: object) -> None:
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttcir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["dxil"] = lambda src, metadata: self.make_dxil(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)
