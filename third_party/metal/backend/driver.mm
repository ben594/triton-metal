#include "torch/csrc/autograd/python_variable.h"
#include <ATen/native/mps/OperationUtils.h>
#import <Metal/Metal.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_metal_buffer, m) {
  m.def("get_mtl_buffer", [](py::object obj) -> uintptr_t {
    at::Tensor t = THPVariable_Unpack(obj.ptr());
    id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
    return reinterpret_cast<uintptr_t>(buf);
  });
}
