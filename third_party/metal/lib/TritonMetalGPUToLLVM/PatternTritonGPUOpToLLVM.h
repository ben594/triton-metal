#ifndef TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
#define TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::metal {
void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);
} // namespace mlir::triton::metal

#endif // TRITON_THIRD_PARTY_METAL_LIB_TRITONMETALGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_