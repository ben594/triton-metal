#ifndef TRITON_METALGPU_DIALECT
#define TRITON_METALGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonMetalGPU_Dialect : Dialect {
  let name = "metalg";
  let cppNamespace = "::mlir::triton::metalgpu";

  let description = [{
    TritonMetalGPU Dialect hosts metal specific ops at TritonGPU abstraction level.
  }];

  let dependentDialects = ["triton::TritonDialect"];

  let useDefaultAttributePrinterParser = 1;
}

#endif
