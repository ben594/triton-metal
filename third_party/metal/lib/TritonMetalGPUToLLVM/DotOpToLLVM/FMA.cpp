#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {

struct DotIntrinsic {
  int vectorSize;
  Type outElemTy;
  StringRef intrinsicName;
  SmallVector<Value> additionalArgs;
};

class MetalFMAVectorMultiplier : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;
  DotIntrinsic intrinsic;

  DotIntrinsic chooseIntrinsic(DotOp op) {
    auto aOpTy = cast<RankedTensorType>(op.getA().getType());
    auto aElemTy = aOpTy.getElementType();
    auto bOpTy = cast<RankedTensorType>(op.getB().getType());
    auto bElemTy = bOpTy.getElementType();
    assert(aElemTy == bElemTy && "a and b must have the same element type");
    auto dOpTy = cast<RankedTensorType>(op.getD().getType());
    auto dElemTy = dOpTy.getElementType();
    assert(aElemTy.isIntOrFloat() && !aElemTy.isIntOrIndex());
    DotIntrinsic chosenOp;
    chosenOp.vectorSize = 1;
    chosenOp.additionalArgs = {};
    // f16 inputs, f32 accumulator
    if (aElemTy.isF16() && dElemTy.isF32()) {
      chosenOp.outElemTy = f32_ty;
      chosenOp.intrinsicName = "llvm.fmuladd.f32";
      return chosenOp;
    }
    assert(aElemTy == dElemTy);
    chosenOp.outElemTy = aElemTy;
    if (aElemTy.isF64())
      chosenOp.intrinsicName = "llvm.fmuladd.f64";
    else if (aElemTy.isF32())
      chosenOp.intrinsicName = "llvm.fmuladd.f32";
    else if (aElemTy.isF16())
      chosenOp.intrinsicName = "llvm.fmuladd.f16";
    else
      llvm_unreachable("unsupported elem type for Metal FMA dot");
    return chosenOp;
  }

  Value packOperand(ArrayRef<Value> scalarValues, int firstElemPos,
                    unsigned vectorSize) {
    if (vectorSize == 1)
      return scalarValues[firstElemPos];
    auto elemTy = scalarValues[firstElemPos].getType();
    auto vecTy = vec_ty(elemTy, vectorSize);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value vec = b.undef(vecTy);
    for (int elem = 0; elem < vectorSize; ++elem) {
      int elemPos = firstElemPos + elem;
      vec =
          b.insert_element(vecTy, vec, scalarValues[elemPos], b.i32_val(elem));
    }
    if (elemTy.isInteger(8)) {
      assert(vectorSize == 4);
      vec = b.bitcast(vec, i32_ty);
    }
    return vec;
  }

  Value generateDotInstr(Value a, Value b, Value c) {
    auto opBuilder = TritonLLVMOpBuilder(loc, rewriter);
    // cast inputs if they don't match output type (e.g. f16 -> f32)
    if (a.getType() != intrinsic.outElemTy)
      a = opBuilder.fpext(intrinsic.outElemTy, a);
    if (b.getType() != intrinsic.outElemTy)
      b = opBuilder.fpext(intrinsic.outElemTy, b);
    SmallVector<Value> args{a, b, c};
    args.append(intrinsic.additionalArgs.begin(),
                intrinsic.additionalArgs.end());
    auto d = LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, intrinsic.intrinsicName, intrinsic.outElemTy, args);
    return d.getResult(0);
  }

public:
  MetalFMAVectorMultiplier(ConversionPatternRewriter &rewriter, DotOp op)
      : rewriter(rewriter), loc(op.getLoc()), intrinsic(chooseIntrinsic(op)) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto kSize = a.size();
    assert(b.size() == kSize);
    Value accum = c;
    for (int k = 0; k < kSize; k += intrinsic.vectorSize) {
      auto aOp = packOperand(a, k, intrinsic.vectorSize);
      auto bOp = packOperand(b, k, intrinsic.vectorSize);
      accum = generateDotInstr(aOp, bOp, accum);
    }
    return accum;
  }
};
} // namespace

namespace mlir::triton::metal {
LogicalResult convertMetalFMADot(DotOp op, DotOp::Adaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  MetalFMAVectorMultiplier multiplier(rewriter, op);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
} // namespace mlir::triton::metal