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

    // try to use air intrinsic ops
    {
      if (aElemTy.isF16() && dElemTy.isF16()) {
        chosenOp.vectorSize = 4;
        chosenOp.outElemTy = f16_ty;
        chosenOp.intrinsicName = "air.dot.v4f16";
        return chosenOp;
      }
      if (aElemTy.isF32() && dElemTy.isF32()) {
        chosenOp.vectorSize = 4;
        chosenOp.outElemTy = f32_ty;
        chosenOp.intrinsicName = "air.dot.v4f32";
        return chosenOp;
      }
    }

    chosenOp.vectorSize = 1;
    chosenOp.additionalArgs = {};
    // f16 inputs, f32 accumulator: cast via air.convert and then use air.dot.v4f32
    if (aElemTy.isF16() && dElemTy.isF32()) {
      chosenOp.vectorSize = 4;
      chosenOp.outElemTy = f32_ty;
      chosenOp.intrinsicName = "air.dot.v4f32";
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

  // cast <4 x f16> vec to <4 x f32> using air.convert
  Value upcastF16ToF32Vec(Value v) {
    auto outTy = vec_ty(f32_ty, 4);
    auto funcType = LLVM::LLVMFunctionType::get(outTy, {v.getType()});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    auto funcOp = appendOrGetExternFuncOp(
        rewriter, parentOp, "air.convert.f.v4f32.f.v4f16", funcType);
    return LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{v})
        .getResult();
  }

  Value generateDotInstr(Value a, Value b, Value c) {
    if (intrinsic.intrinsicName.starts_with("air.")) {
      // air.dot has no accumulator arg, so accumulate with fadd
      // if inputs are f16 but output is f32, cast inputs
      if (a.getType() == vec_ty(f16_ty, 4) && intrinsic.outElemTy == f32_ty) {
        a = upcastF16ToF32Vec(a);
        b = upcastF16ToF32Vec(b);
      }
      auto aType = a.getType();
      auto funcType =
          LLVM::LLVMFunctionType::get(intrinsic.outElemTy, {aType, aType});
      Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
      auto funcOp = appendOrGetExternFuncOp(rewriter, parentOp,
                                            intrinsic.intrinsicName, funcType);
      auto opBuilder = TritonLLVMOpBuilder(loc, rewriter);
      Value dot =
          LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{a, b})
              .getResult();
      return opBuilder.fadd(dot, c);
    }

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