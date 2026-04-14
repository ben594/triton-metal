#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using mlir::triton::ModuleAxisInfoAnalysis;
using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

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

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type type = val.getType();

  assert(type == f32_ty && "only f32 supported for now");

  // TODO don't hardcode f32
  StringRef funcName = "air.simd_shuffle_xor.f32";
  Type funcType = getFunctionType(type, ValueRange{val, i});
  Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parentOp, funcName, funcType);

  Value result =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{val, i})
          .getResult();
  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i16_val(i));
}
} // namespace mlir::LLVM::metal