add_triton_library(TritonMetalGPUToLLVM
    FuncOpToLLVM.cpp
    MembarUtility.cpp
    MemoryOpToLLVM.cpp
    TargetInfo.cpp
    TritonGPUToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    Utility.cpp
    GPUIdxOpsToLLVM.cpp
    SPMDOpToLLVM.cpp

    DEPENDS
    TritonMetalGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonMetalAnalysis
)