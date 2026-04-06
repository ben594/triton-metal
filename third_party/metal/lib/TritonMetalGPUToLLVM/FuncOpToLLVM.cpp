

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {
struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    handleArgPtrDatatype(funcOp, newFuncOp);

    if (triton::isKernel(funcOp)) {
      newFuncOp.setLinkage(LLVM::Linkage::External);

      // set func attributes
      {
        SmallVector<Attribute> funcAttrs;
        auto addStr = [&](StringRef s) {
          funcAttrs.push_back(rewriter.getStringAttr(s));
        };
        addStr("nounwind");
        addStr("no-builtins");
        newFuncOp.setPassthroughAttr(ArrayAttr::get(ctx, funcAttrs));

        newFuncOp.setUnnamedAddr(LLVM::UnnamedAddr::Local);
      }

      // pass thread/simd/group idxs as extra i32 params to kernel
      // TODO set metadata and handle multiple dims
      auto i32Type = IntegerType::get(ctx, 32);

      SmallVector<DictionaryAttr> argAttrs;
      newFuncOp.getAllArgAttrs(argAttrs);

      auto llvmFuncType = newFuncOp.getFunctionType();
      SmallVector<Type> params(llvmFuncType.getParams());
      params.push_back(i32Type);
      params.push_back(i32Type);
      params.push_back(i32Type);
      newFuncOp.setFunctionType(
          LLVM::LLVMFunctionType::get(llvmFuncType.getReturnType(), params));

      // first entry block receives args from function params
      // so need to add additional params to first entry block
      auto &region = newFuncOp.getBody();
      auto loc = funcOp.getLoc();
      region.addArgument(i32Type, loc);
      region.addArgument(i32Type, loc);
      region.addArgument(i32Type, loc);
      auto noundef =
          rewriter.getNamedAttr("llvm.noundef", rewriter.getUnitAttr());
      auto argAttr = DictionaryAttr::get(ctx, {noundef});
      argAttrs.push_back(argAttr);
      argAttrs.push_back(argAttr);
      argAttrs.push_back(argAttr);
      newFuncOp.setAllArgAttrs(argAttrs);
    } else {
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }

    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::metal::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, targetInfo, benefit);
}