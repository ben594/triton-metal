#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  explicit GetNumProgramsOpConversion(LLVMTypeConverter &typeConverter,
                                      const metal::TargetInfo &targetInfo,
                                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetNumProgramsOp>(typeConverter,
                                                         benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value numPrograms = targetInfo.numPrograms(
        rewriter, op->getLoc(), op->getParentOfType<ModuleOp>(), op.getAxis());
    rewriter.replaceOp(op, numPrograms);
    return success();
  }

private:
  const metal::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::metal::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const metal::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, targetInfo, benefit);
}
