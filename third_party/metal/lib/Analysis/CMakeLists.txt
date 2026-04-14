add_triton_library(TritonMetalAnalysis
  MetalGPUAllocation.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  TritonIR
  TritonGPUIR
)

target_include_directories(TritonMetalAnalysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
