#include "Analysis/MetalGPUAllocation.h"
#include "MembarUtility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonMetalGPUToLLVM/Passes.h"
#include "metal/lib/TritonMetalGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONMETALGPUTOLLVM
#include "TritonMetalGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {
class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();

    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<triton::instrument::TritonInstrumentDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct UnrealizedCastToLoadPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  UnrealizedCastToLoadPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<UnrealizedConversionCastOp>(ctx, benefit) {};

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // handle ptr to scalar casts for scalar kernel args
    auto inputs = castOp.getInputs();
    if (inputs.size() != 1) {
      return failure();
    }
    if (!mlir::isa<LLVM::LLVMPointerType>(inputs[0].getType())) {
      return failure();
    }
    // only handle scalar LLVM output types (int/float)
    // TODO can check if input is a kernel arg?
    auto resultType = castOp.getType(0);
    if (!mlir::isa<IntegerType, FloatType>(resultType))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(castOp, castOp.getType(0),
                                              inputs[0]);
    return success();
  }
};

struct ConvertTritonMetalGPUToLLVM
    : public triton::impl::ConvertTritonMetalGPUToLLVMBase<
          ConvertTritonMetalGPUToLLVM> {

  explicit ConvertTritonMetalGPUToLLVM(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    metal::TargetInfo targetInfo(this->arch.getValue());

    // Allocate shared memory and set barrier
    {
      ModuleAllocation allocation(mod,
                                  metal::MetalAllocationAnalysisScratchSizeFn);

      ModuleMembarAnalysis membarPass(&allocation,
                                      mlir::triton::metal::membarFilter);
      membarPass.run();
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    // TODO skip shared memory for now just to get vector add example working

    // lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::metal::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    {
      RewritePatternSet cleanupPatterns(context);
      cleanupPatterns.add<UnrealizedCastToLoadPattern>(context, benefit);
      if (failed(applyPatternsGreedily(mod, std::move(cleanupPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);

    metal::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    metal::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    metal::populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);

    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);

    // this handles program id
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);

    // this handles num programs
    metal::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                       benefit);

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::triton::metal::populateGPUIdxOpsConversionPattern(
        typeConverter, patterns, targetInfo, benefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    // "or disjoint" op seems to not be supported by metal-as
    mod.walk([](LLVM::OrOp op) { op.setIsDisjoint(false); });
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMetalGPUToLLVMPass(StringRef targetArch) {
  return std::make_unique<ConvertTritonMetalGPUToLLVM>(targetArch);
}

} // namespace mlir::triton