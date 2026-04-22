#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonmetal-allocate-smem-for-simdgroup-matmul"

namespace mlir {
#define GEN_PASS_DEF_TRITONMETALGPUALLOCATESMEMFORSIMDGROUPMATMUL
#include "TritonMetalGPUTransforms/Passes.h.inc"

struct TritonMetalGPUAllocateSmemForSimdgroupMatmulPass
    : impl::TritonMetalGPUAllocateSmemForSimdgroupMatmulBase<
          TritonMetalGPUAllocateSmemForSimdgroupMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    mod.walk([&](tt::DotOp dotOp) {
      auto *ctx = dotOp.getContext();
      auto loc = dotOp.getLoc();

      auto aTensorTy = cast<RankedTensorType>(dotOp.getA().getType());
      auto bTensorTy = cast<RankedTensorType>(dotOp.getB().getType());
      auto retType = cast<RankedTensorType>(dotOp.getResult().getType());

      // result must have blocked encoding
      if (!isa<ttg::BlockedEncodingAttr>(retType.getEncoding()))
        return;

      auto sharedMemSpace = ttg::SharedMemorySpaceAttr::get(ctx);
      auto cgaLayout =
          ttg::CGAEncodingAttr::get1CTALayout(ctx, /*rank=*/retType.getRank());

      // no padding, plain row-major layout
      SmallVector<std::pair<unsigned, unsigned>> noPadding;
      SmallVector<unsigned> rowMajor = {1, 0};

      auto makeSharedTy = [&](ArrayRef<int64_t> shape, Type elemTy) {
        auto enc = ttg::PaddedSharedEncodingAttr::get(ctx, noPadding, rowMajor,
                                                      shape, cgaLayout);
        return ttg::MemDescType::get(shape, elemTy, enc, sharedMemSpace,
                                     /*mutableMemory=*/true);
      };

      // insert LocalAllocOp ops before dot op
      builder.setInsertionPoint(dotOp);
      ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(aTensorTy.getShape(), aTensorTy.getElementType()));
      ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(bTensorTy.getShape(), bTensorTy.getElementType()));
      ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(retType.getShape(), retType.getElementType()));
    });
  }
};

} // namespace mlir
