#ifndef TRITON_THIRD_PARTY_METAL_INCLUDE_TRITONMETALGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_METAL_INCLUDE_TRITONMETALGPUTRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "third_party/metal/include/Dialect/TritonMetalGPU/IR/Dialect.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonMetalGPUTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir::triton::metalgpu {} // namespace mlir::triton::metalgpu

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonMetalGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_METAL_INCLUDE_TRITONMETALGPUTRANSFORMS_PASSES_H_
