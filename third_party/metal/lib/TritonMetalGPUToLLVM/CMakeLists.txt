add_triton_library(TritonMetalGPUToLLVM
    FuncOpToLLVM.cpp
    TargetInfo.cpp
    TritonGPUToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    Utility.cpp
    GPUThreadIdOpToLLVM.cpp

    DEPENDS
    TritonMetalGPUConversionPassIncGen
)