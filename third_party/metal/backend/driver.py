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
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MetalUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # only init once
        if hasattr(self, "_initialized"):
            return

        import Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
        self._initialized = True

    def load_binary(self, kernel_name, metallib_bytes, shared_mem, device):
        """
        Load metallib binary
        Inputs: name, kernel (bytes), metadata.shared, device
        Needs to return: module, function, n_regs, n_spills, n_max_threads
        """
        import objc

        data = objc.lookUpClass("NSData").dataWithBytes_length_(metallib_bytes, len(metallib_bytes))

        lib, err = device.newLibraryWithData_error_(data, None)
        assert lib is not None, f"failed to load metal lib: {err}"

        func = lib.newFunctionWithName_(kernel_name)
        assert func is not None, f"kernel '{kernel_name}' not found in lib"

        pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
        assert pipeline is not None, f"failed to create pipeline state: {err}"

        # TODO is this correct?
        n_max_threads = pipeline.maxTotalThreadsPerThreadgroup()

        # TODO verify can return 0 for n_regs and n_spills
        return lib, pipeline, 0, 0, n_max_threads

    def unload_module(self):
        raise NotImplementedError

    def get_device_properties(self, device):
        raise NotImplementedError


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
