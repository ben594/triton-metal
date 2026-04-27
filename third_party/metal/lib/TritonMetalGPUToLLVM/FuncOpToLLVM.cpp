

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
    // when adding implicit stride args for tensors, don't want to include the
    // amended scratch ptrs
    unsigned numTrueUserArgs = funcOp.getNumArguments();

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
      auto funcType = newFuncOp.getFunctionType();
      SmallVector<Type> params(funcType.getParams());
      SmallVector<Type> origParams(funcType.getParams());
      SmallVector<DictionaryAttr> origArgAttrs;
      newFuncOp.getAllArgAttrs(origArgAttrs);

      // build up new argAttrs
      SmallVector<DictionaryAttr> newArgAttrs;
      auto origNumUserArgs = newFuncOp.getNumArguments();

      // for tensor args, add implicit stride arguments (ptr to vector)
      SmallVector<Type> paramsWithImplicitStrides;
      {
        auto numUserArgs = newFuncOp.getNumArguments();
        auto globalPtrType = LLVM::LLVMPointerType::get(ctx, 1);

        for (unsigned i = 0; i < numUserArgs; i++) {
          if (mlir::isa<LLVM::LLVMPointerType>(params[i]) &&
              i < numTrueUserArgs) {
            // tensor arg -> add implicit stride arg
            auto strideArgType = globalPtrType; // ptr to vector of strides
            paramsWithImplicitStrides.push_back(params[i]);
            paramsWithImplicitStrides.push_back(strideArgType);
            newArgAttrs.push_back(origArgAttrs[i]);
            NamedAttrList strideAttrs;
            strideAttrs.set(
                "metal.implicit_stride_for",
                rewriter.getI32IntegerAttr(i)); // tracks original ptr arg index
            newArgAttrs.push_back(DictionaryAttr::get(ctx, strideAttrs));
          } else {
            paramsWithImplicitStrides.push_back(params[i]);
            newArgAttrs.push_back(origArgAttrs[i]);
          }
        }
      }

      newFuncOp.setFunctionType(LLVM::LLVMFunctionType::get(
          funcType.getReturnType(), paramsWithImplicitStrides));

      // convert scalar user args to ptr addrspace(1)
      {
        auto numUserArgsWithImplicitStrides = newFuncOp.getNumArguments();
        auto globalPtrType = LLVM::LLVMPointerType::get(ctx, 1);
        for (unsigned i = 0; i < numUserArgsWithImplicitStrides; i++) {
          if (!mlir::isa<LLVM::LLVMPointerType>(paramsWithImplicitStrides[i])) {
            paramsWithImplicitStrides[i] = globalPtrType; // scalar -> ptr
            NamedAttrList attrs;
            attrs.set("llvm.noundef", rewriter.getUnitAttr());
            attrs.set("llvm.nocapture", rewriter.getUnitAttr());
            attrs.set("llvm.readonly", rewriter.getUnitAttr());
            attrs.set("llvm.dereferenceable",
                      rewriter.getIntegerAttr(IntegerType::get(ctx, 64), 4));
            newArgAttrs[i] = DictionaryAttr::get(ctx, attrs);
          }
        }

        newFuncOp.setFunctionType(LLVM::LLVMFunctionType::get(
            funcType.getReturnType(), paramsWithImplicitStrides));

        // modify first block
        unsigned insertOffset = 0;
        auto &firstBlock = newFuncOp.getBody().front();
        rewriter.setInsertionPointToStart(&firstBlock);

        // use numTrueUserArgs here to not modify the amended scratch ptr args
        for (unsigned i = 0; i < numTrueUserArgs; i++) {
          auto arg = firstBlock.getArgument(
              i + insertOffset); // account for prior insertions
          if (!mlir::isa<LLVM::LLVMPointerType>(origParams[i])) {
            arg.setType(globalPtrType);
          } else {
            unsigned insertIdx = i + 1 + insertOffset;
            // insert for newly added implicit stride args
            firstBlock.insertArgument(insertIdx, globalPtrType, arg.getLoc());
            insertOffset++;
          }
        }
      }

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

      // add kNumI32ExtraArgs i32 args (num_programs, thread_idx, simdgroup_idx,
      // threadgroup_idx)
      // see MetalKernelArgs.h for layout
      for (int i = 0; i < mlir::triton::metal::kNumI32ExtraArgs; ++i) {
        paramsWithImplicitStrides.push_back(i32Type);
      }
      newFuncOp.setFunctionType(LLVM::LLVMFunctionType::get(
          funcType.getReturnType(), paramsWithImplicitStrides));

      // first entry block receives args from function params
      // so need to add additional params to first entry block
      auto &region = newFuncOp.getBody();
      auto loc = funcOp.getLoc();
      auto noundef =
          rewriter.getNamedAttr("llvm.noundef", rewriter.getUnitAttr());
      auto argAttr = DictionaryAttr::get(ctx, {noundef});
      for (int i = 0; i < mlir::triton::metal::kNumI32ExtraArgs; ++i) {
        region.addArgument(i32Type, loc);
        newArgAttrs.push_back(argAttr);
      }
      newFuncOp.setAllArgAttrs(newArgAttrs);
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