#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace mlir::triton::metal {
LogicalResult convertMetalFMADot(triton::DotOp op,
                                 triton::DotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::metal

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return metal::convertMetalFMADot(op, adaptor, getTypeConverter(),
                                       rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

namespace mlir::triton::metal {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::metal