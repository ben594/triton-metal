#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonMetalGPUAccelerateMatmul : Pass<"accelerate-metal-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

#endif // TRITONGPU_PASSES