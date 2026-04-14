#ifndef TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::metal {

// Filter function for Metal backend to filter unnecessary barriers
// during membar analysis
//
// Currently no-op (always false)
// Add cases as Metal-specific async/pipeline ops are introduced
bool membarFilter(Operation *op1, Operation *op2, bool op1IsRead,
                  bool op2IsRead, Allocation *allocation);

} // namespace mlir::triton::metal

#endif // TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_MEMBARUTILITY_H_
