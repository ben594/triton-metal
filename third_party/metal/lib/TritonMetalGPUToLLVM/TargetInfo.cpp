#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir::triton::metal {

bool TargetInfo::supportMaximumMinimum() const { return true; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  if (triton::gpu::lookupNumCTAs(&rewriter.getInsertionBlock()->front()) == 1)
    return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  llvm_unreachable("getClusterCTAId not implemented for CTA > 1");
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm_unreachable("not implemented");
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(targets);
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter) const {
  llvm_unreachable("not implemented");
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  llvm_unreachable("not implemented");
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "Metal GPU does not support cross-CTA shared memory transfers");
  }
  mlir::LLVM::metal::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "Metal GPU does not support cross-CTA shared memory transfers");
  }
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::metal::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::metal::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm_unreachable("not implemented");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("not implemented");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm_unreachable("not implemented");
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm_unreachable("not implemented");
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  auto func = rewriter.getInsertionBlock()
                  ->getParent()
                  ->getParentOfType<LLVM::LLVMFuncOp>();
  unsigned numArgs = func.getNumArguments();
  if (axis == ProgramIDDim::X) {
    // plan to pass threadgroup idx in grid as last arg
    return func.getArgument(numArgs - 1);
  }
  llvm_unreachable("Only X axis supported for now");
}

Value TargetInfo::numPrograms(RewriterBase &rewriter, Location loc,
                              ModuleOp moduleOp, ProgramIDDim axis) const {
  auto func = rewriter.getInsertionBlock()
                  ->getParent()
                  ->getParentOfType<LLVM::LLVMFuncOp>();
  unsigned numArgs = func.getNumArguments();
  if (axis == ProgramIDDim::X) {
    // plan to pass number of threadgroups in grid as arg
    return func.getArgument(numArgs - 4);
  }
  llvm_unreachable("Only X axis supported for now");
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  llvm_unreachable("not implemented");
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  llvm_unreachable("not implemented");
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  llvm_unreachable("not implemented");
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  llvm_unreachable("not implemented");
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  llvm_unreachable("not implemented");
}

bool TargetInfo::supportVectorizedAtomics() const {
  llvm_unreachable("not implemented");
}

} // namespace mlir::triton::metal