#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

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

    int64_t dotIdx = 0; // support multiple dot ops per function

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

      StringAttr kOffset = StringAttr::get(ctx, "offset");
      StringAttr kBlock = StringAttr::get(ctx, "block");
      StringAttr dim0 = StringAttr::get(ctx, "dim0");
      StringAttr dim1 = StringAttr::get(ctx, "dim1");

      auto makeSharedTy = [&](ArrayRef<int64_t> shape, Type elemTy) {
        int64_t M = shape[0], N = shape[1];
        unsigned alignment = elemTy.getIntOrFloatBitWidth() / 8;

        // row major: offset = row * N + col
        std::vector<std::vector<int32_t>> offsetBases;
        for (int64_t i = 1; i < N; i *= 2)
          offsetBases.push_back({0, (int32_t)i});
        for (int64_t i = 1; i < M; i *= 2)
          offsetBases.push_back({(int32_t)i, 0});

        // single CTA, 0 basis vectors
        std::vector<std::vector<int32_t>> blockBases = {};

        triton::LinearLayout ll({{kOffset, offsetBases}, {kBlock, blockBases}},
                                {{dim0, (int32_t)M}, {dim1, (int32_t)N}},
                                /*requireSurjective=*/true);
        auto enc = ttg::SharedLinearEncodingAttr::get(ctx, ll, alignment);
        return ttg::MemDescType::get(shape, elemTy, enc, sharedMemSpace,
                                     /*mutableMemory=*/true);
      };

      // allocate smem right before dot op
      // previously hoisted this outside of the loop, but that overlapped with
      // convert layout scratch space and may take up too much smem
      builder.setInsertionPoint(dotOp);

      auto aAlloc = ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(aTensorTy.getShape(), aTensorTy.getElementType()));
      aAlloc->setAttr("metal.dot_smem", StringAttr::get(ctx, "A"));
      aAlloc->setAttr(
          "metal.dot_idx",
          IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));
      auto bAlloc = ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(bTensorTy.getShape(), bTensorTy.getElementType()));
      bAlloc->setAttr("metal.dot_smem", StringAttr::get(ctx, "B"));
      bAlloc->setAttr(
          "metal.dot_idx",
          IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));

      dotOp->setAttr("metal.dot_idx",
                     IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));

      // store A and B in smem before the dot op
      // when lowering the dot op, can load from smem
      auto aSmemStore = ttg::LocalStoreOp::create(builder, loc, dotOp.getA(),
                                                  aAlloc.getResult());
      auto bSmemStore = ttg::LocalStoreOp::create(builder, loc, dotOp.getB(),
                                                  bAlloc.getResult());

      // dealloc right after the dot op to bound liveness tightly
      builder.setInsertionPointAfter(dotOp);
      ttg::LocalDeallocOp::create(builder, loc, aAlloc->getResult(0));
      ttg::LocalDeallocOp::create(builder, loc, bAlloc->getResult(0));

      dotIdx++;
    });
  }
};

} // namespace mlir
