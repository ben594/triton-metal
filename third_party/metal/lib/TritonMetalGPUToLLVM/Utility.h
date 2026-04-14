#include "triton/Analysis/AxisInfo.h"

using mlir::triton::ModuleAxisInfoAnalysis;

namespace mlir::LLVM::metal {
// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
// forceNoAliasAsyncLoads=true adds alias information to the llvm.load to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal);

// Stores to shared or global memory with predication.
// forceNoAliasAsyncLoads=true adds alias information to the llvm.store to
// signal its not aliasing with any AsyncCopyGlobalToLocal/BufferLoadToLocal to
// avoid conservative waits. See `addLocalLoadNoAliasScope` for more details
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred);

// Get contiguity for a tensor pointer `ptr`
unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);
// Determine the vector size of a tensor of pointers
unsigned getVectorSize(Value ptr,
                       triton::ModuleAxisInfoAnalysis &axisAnalysisPass);

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
} // namespace mlir::LLVM::metal