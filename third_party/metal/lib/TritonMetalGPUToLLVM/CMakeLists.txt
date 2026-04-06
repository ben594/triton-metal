add_triton_library(TritonMetalGPUToLLVM
    FuncOpToLLVM.cpp
    TargetInfo.cpp
    TritonGPUToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    Utility.cpp
    GPUIdxOpsToLLVM.cpp

    DEPENDS
    TritonMetalGPUConversionPassIncGen
)