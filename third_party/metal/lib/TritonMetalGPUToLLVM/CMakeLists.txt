add_triton_library(TritonMetalGPUToLLVM
    FuncOpToLLVM.cpp
    TargetInfo.cpp
    TritonGPUToLLVM.cpp
    ElementwiseOpToLLVM.cpp

    DEPENDS
    TritonMetalGPUConversionPassIncGen
)