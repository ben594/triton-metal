#ifndef TRITONMETAL_ANALYSIS_METALGPU_ALLOCATION_H
#define TRITONMETAL_ANALYSIS_METALGPU_ALLOCATION_H

#include "mlir/IR/Operation.h"

namespace mlir::triton::metal {

unsigned MetalAllocationAnalysisScratchSizeFn(Operation *op);

} // namespace mlir::triton::metal

#endif // TRITONMETAL_ANALYSIS_METALGPU_ALLOCATION_H
