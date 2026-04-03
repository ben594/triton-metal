#include "TritonMetalGPUToLLVM/Passes.h"
#include "mlir/Pass/PassManager.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
const char *const metalTargetTriple = "metal";

void init_triton_metal_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;
  m.def("add_to_llvmir", [](mlir::PassManager &pm, const std::string &arch) {
    pm.addPass(mlir::triton::createConvertTritonMetalGPUToLLVMPass(arch));
  });
}

} // namespace

void init_triton_metal(py::module &&m) {
  m.doc() = "Python bindings to the Metal Triton backend";

  auto passes = m.def_submodule("passes");
  init_triton_metal_passes_ttgpuir(passes.def_submodule("ttgpuir"));
}
