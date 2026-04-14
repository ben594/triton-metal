#include "Analysis/MetalGPUAllocation.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::metal {

unsigned MetalAllocationAnalysisScratchSizeFn(Operation *op) {
  return defaultAllocationAnalysisScratchSizeFn(op);
}

} // namespace mlir::triton::metal
