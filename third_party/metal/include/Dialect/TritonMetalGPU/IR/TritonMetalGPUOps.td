#ifndef TRITON_METALGPU_OPS
#define TRITON_METALGPU_OPS

class TT_MetalGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<TritonMetalGPU_Dialect, mnemonic, !listconcat(traits, [])>;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

def SimdgroupMMAOp : TT_METALGPU_Op<"simdgroup_mma", [Pure]> {
  let summary = "simdgroup matrix multiply-accumulate";
  let description = [{
    This op represents a simdgroup MMA instruction.
  }];

  let arguments = (ins
    TT_Ptr:$a_ptr,
    TT_Ptr:$b_ptr,
    TT_Ptr:$c_ptr,
    TT_Ptr:$d_ptr,
    I32:$strideA,
    I32:$strideB,
    I32:$strideC,
    I32:$strideD
  );
  
  let assemblyFormat = [{
    $a_ptr `,` $b_ptr `,` $c_ptr `,` $d_ptr `,`
    $strideA `,` $strideB `,` $strideC `,` $strideD
  }];
}

#endif // TRITON_METALGPU_OPS