#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {
struct GPUThreadIdOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
  GPUThreadIdOpConversion(LLVMTypeConverter &converter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp threadIdOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dim = threadIdOp.getDimension();
    if (dim != mlir::gpu::Dimension::x) {
      llvm_unreachable("Only X axis supported for now");
    }

    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();

    // plan to pass thread idx in grid as second to last arg
    Value threadIdx = func.getArgument(numArgs - 2);
    auto idxType = rewriter.getIndexType();
    rewriter.replaceOp(threadIdOp, threadIdx);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::metal::populateGPUThreadIDOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GPUThreadIdOpConversion>(typeConverter, targetInfo, benefit);
}