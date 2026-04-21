#ifndef TRITON_METALGPU_OPS
#define TRITON_METALGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

include "TritonMetalGPUDialect.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


class TT_MetalGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<TritonMetalGPU_Dialect, mnemonic, !listconcat(traits, [])>;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

def SimdgroupMMAOp : TT_MetalGPU_Op<"simdgroup_mma", [Pure]> {
  let summary = "simdgroup matrix multiply-accumulate";
  let description = [{
    This op represents a simdgroup MMA instruction.
  }];

  let arguments = (ins
    Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$a_ptr,
    Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$b_ptr,
    Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$c_ptr,
    Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$d_ptr,
    I32:$strideA,
    I32:$strideB,
    I32:$strideC,
    I32:$strideD
  );
  
  let assemblyFormat = [{
    $a_ptr `,` $b_ptr `,` $c_ptr `,` $d_ptr `,`
    $strideA `,` $strideB `,` $strideC `,` $strideD
    attr-dict `:` type(operands)
  }];
}

#endif // TRITON_METALGPU_OPS