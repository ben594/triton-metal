#include "MembarUtility.h"

namespace mlir::triton::metal {

bool membarFilter(Operation *op1, Operation *op2, bool /*op1IsRead*/,
                  bool /*op2IsRead*/, Allocation * /*allocation*/) {
  // TODO fill in as needed to decide when to skip barriers
  return false;
}

} // namespace mlir::triton::metal
