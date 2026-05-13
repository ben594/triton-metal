add_triton_library(TritonMetalGPUIR
  Dialect.cpp

  DEPENDS
  TritonMetalGPUTableGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
