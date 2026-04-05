#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using mlir::triton::ModuleAxisInfoAnalysis;

namespace mlir::LLVM::metal {
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal) {
  // unconditional load and then select
  // if pred=1 (unmasked), LLVM will optimize away select
  Value loadedVal =
      LLVM::LoadOp::create(rewriter, loc, elemTy, ptr).getResult();
  return LLVM::SelectOp::create(rewriter, loc, pred, loadedVal, falseVal)
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred) {
  auto *curBlock = rewriter.getInsertionBlock();

  // split current block at store op
  auto *afterBlock = curBlock->splitBlock(rewriter.getInsertionPoint());

  // create new block between original block and afterBlock
  auto *storeBlock = rewriter.createBlock(afterBlock);

  // branch to storeBlock if pred=1
  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, storeBlock, afterBlock);

  // emit store
  rewriter.setInsertionPointToStart(storeBlock);
  LLVM::StoreOp::create(rewriter, loc, val, ptr);
  LLVM::BrOp::create(rewriter, loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
}

unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  return axisAnalysisPass.getContiguity(ptr);
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto contiguity = getContiguity(ptr, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}
} // namespace mlir::LLVM::metal