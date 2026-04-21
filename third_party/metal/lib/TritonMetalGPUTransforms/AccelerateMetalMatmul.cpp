#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonmetal-accelerate-matmul"

namespace mlir {
namespace {
Value findBasePtr(PatternRewriter &rewriter, Value val, tt::DotOp dotOp) {
  auto convertLayoutOp = val.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!convertLayoutOp) {
    return {};
  }

  auto ttLoadOp = convertLayoutOp.getSrc().getDefiningOp<tt::LoadOp>();
  if (!ttLoadOp) {
    return {};
  }

  auto ptr = ttLoadOp.getPtr();

  while (auto addPtrOp = ptr.getDefiningOp<tt::AddPtrOp>()) {
    ptr = addPtrOp.getPtr();
  }

  auto ttSplatOp = ptr.getDefiningOp<tt::SplatOp>();
  if (!ttSplatOp) {
    return {};
  }

  return ttSplatOp.getSrc();
}

class DotToMetalMatmul : public OpRewritePattern<tt::DotOp> {
public:
  DotToMetalMatmul(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    auto elemType = oldRetType.getElementType();
    if (!elemType.isF32()) {
      return rewriter.notifyMatchFailure(dotOp, "expected f32 element type");
    }
    auto retShape = oldRetType.getShape();
    int numWarps = ttg::lookupNumWarps(dotOp);

    // find base ptrs for A and B
    Value aBasePtr = findBasePtr(rewriter, dotOp.getA(), dotOp);
    Value bBasePtr = findBasePtr(rewriter, dotOp.getB(), dotOp);
    if (!aBasePtr || !bBasePtr) {
      return rewriter.notifyMatchFailure(
          dotOp, "failed to find base ptrs for A and B");
    }

    llvm::errs() << "a base ptr: " << aBasePtr << "\n";
    llvm::errs() << "b base ptr: " << bBasePtr << "\n";

    // TODO need to look forward to find the store op that uses dot op
    // accumulated result

    return success();
  }
};
} // namespace

#define GEN_PASS_DEF_TRITONMETALGPUACCELERATEMATMUL
#include "TritonMetalGPUTransforms/Passes.h.inc"

struct TritonMetalGPUAccelerateMatmulPass
    : impl::TritonMetalGPUAccelerateMatmulBase<
          TritonMetalGPUAccelerateMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<DotToMetalMatmul>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  };
};

} // namespace mlir