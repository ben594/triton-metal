import struct
import functools
import triton
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


class MetalUtils:
    pass


class MetalLauncher:
    pass


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

        return torch.device("mps")

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
        pass

    def get_empty_cache_for_benchmark(self):
        import torch

        # TODO copied from other backends, modify if needed
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="mps")

    def clear_cache(self, cache):
        # TODO copied from other backends, modify if needed
        cache.zero_()
