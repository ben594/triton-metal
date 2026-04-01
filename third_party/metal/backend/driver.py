import functools
import os
import struct
import tempfile

import Metal
import objc
import torch
import triton
from triton._C import _metal_buffer
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def ty_to_cpp(ty):
    if ty[0] == "*":
        raise NotImplementedError("Pointer types are not supported in metal ty_to_cpp yet")
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


# Scalar type -> struct.pack format string
_SCALAR_FMT = {
    "i1": "b",
    "i8": "b",
    "i16": "h",
    "i32": "i",
    "i64": "q",
    "u1": "B",
    "u8": "B",
    "u16": "H",
    "u32": "I",
    "u64": "Q",
    "fp16": "e",
    "bf16": "e",
    "fp32": "f",
    "f32": "f",
    "fp64": "d",
}


class MetalUtils:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MetalUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # only init once
        if hasattr(self, "_initialized"):
            return

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.get_cmd_buf()
        self._initialized = True

    def get_cmd_buf(self):
        return objc.objc_object(c_void_p=_metal_buffer.get_command_buffer())

    def load_binary(self, kernel_name, metallib_bytes, shared_mem, device):
        """
        Load metallib binary
        Inputs: name, kernel (bytes), metadata.shared, device
        Needs to return: module, function, n_regs, n_spills, n_max_threads
        """
        from Foundation import NSURL

        with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
            f.write(metallib_bytes)
            tmp_path = f.name

        try:
            url = NSURL.fileURLWithPath_(tmp_path)
            lib, err = self.device.newLibraryWithURL_error_(url, None)
            assert lib is not None, f"failed to load metal lib: {err}"

            func = lib.newFunctionWithName_(kernel_name)
            assert func is not None, f"kernel '{kernel_name}' not found in lib"

            pipeline, err = self.device.newComputePipelineStateWithFunction_error_(func, None)
            assert pipeline is not None, f"failed to create pipeline state: {err}"

            n_max_threads = pipeline.maxTotalThreadsPerThreadgroup()
            return lib, pipeline, 0, 0, n_max_threads
        finally:
            os.unlink(tmp_path)

    def unload_module(self):
        # TODO verify if kernel is garbage collected
        pass

    def get_device_properties(self, device):
        # TODO verify these
        return {
            "max_shared_mem": self.device.maxThreadgroupMemoryLength(),
            "multiprocessor_count": 0,  # n/a
            "mem_clock_rate": 0,  # n/a
            "mem_bus_width": 0,  # n/a
        }


class MetalLauncher:
    def __init__(self, src, metadata):
        self.constants = src.constants if hasattr(src, "constants") else {}
        self.signature = src.signature
        self.num_warps = metadata.num_warps
        self.warp_size = 32  # Apple GPU SIMD group size

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,  # func returned in load_binary
        kernel_metadata,  # packed_metadata from backend pack_metadata
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,  # contains caller args
    ):
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        # stream is MTLCommandQueue, function is MTLComputePipelineState
        cmd_buf = stream
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(function)

        signature_types = list(self.signature.values())
        metal_idx = 0
        for arg, ty in zip(args, signature_types):
            if ty == "constexpr":
                # skip constexpr args
                continue
            if ty[0] == "*":
                assert isinstance(arg, torch.Tensor), f"expected torch tensor for ptr arg, got {type(arg)}"
                assert arg.device.type == "mps", f"expected mps tensor, got {arg.device}"
                assert arg.is_contiguous(), "metal backend requires contiguous tensors"
                # raw_ptr = _metal_buffer.get_mtl_buffer(arg)
                torch.mps.synchronize()
                raw_ptr = arg.storage().data_ptr()
                print(f"raw_ptr: {raw_ptr}")
                print(arg.storage().data_ptr())
                assert raw_ptr != 0, f"MPS tensor returned null MTLBuffer for arg {metal_idx}"
                buf = objc.objc_object(c_void_p=raw_ptr)
                encoder.setBuffer_offset_atIndex_(buf, 0, metal_idx)
            else:
                fmt = _SCALAR_FMT.get(ty)
                assert fmt is not None, f"scalar type {ty} not found"
                packed = struct.pack(fmt, arg)
                device = triton.runtime.driver.active.utils.device
                buf = device.newBufferWithBytes_length_options_(packed, len(packed), Metal.MTLResourceStorageModeShared)
                encoder.setBuffer_offset_atIndex_(buf, 0, metal_idx)
            metal_idx += 1

        threads_per_tg = Metal.MTLSizeMake(self.num_warps * self.warp_size, 1, 1)
        threadgroups = Metal.MTLSizeMake(gridX, gridY, gridZ)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threads_per_tg)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        if cmd_buf.error():
            raise RuntimeError(f"Metal command buffer error: {cmd_buf.error()}")

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class MetalDriver(DriverBase):
    def __init__(self):
        super().__init__()
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher
        self.get_current_device = self._get_current_device
        self.set_current_device = self._set_current_device
        self.get_current_stream = self._get_current_stream

    @staticmethod
    def is_active():
        try:
            import Metal

            return Metal.MTLCreateSystemDefaultDevice() is not None
        except ImportError:
            return False

    @functools.lru_cache()
    def get_current_target(self):
        # SIMD group size = 32
        # TODO can hardcode this?
        # TODO need to specify arch?
        return GPUTarget("metal", 0, 32)

    def get_active_torch_device(self):
        import torch

        return torch.device("mps", 0)

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def _get_current_device(self):
        return self.utils.device

    def _set_current_device(self, device):
        # TODO noop for now, change if needed
        pass

    def _get_current_stream(self, device):
        return self.utils.command_queue

    def get_empty_cache_for_benchmark(self):
        import torch

        # TODO copied from other backends, modify if needed
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="mps")

    def clear_cache(self, cache):
        # TODO copied from other backends, modify if needed
        cache.zero_()

    def get_device_interface(self):
        import torch
        return torch.mps
