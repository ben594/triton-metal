include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonMetal ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_metal.cc LINK_LIBS TritonMetalGPUToLLVM)
  target_link_libraries(TritonMetal PRIVATE Python3::Module pybind11::headers)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  # Find torch paths for the standalone ATen extension
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(TORCH_LIB_DIR "${TORCH_PATH}/lib")

  # use separate library from libtriton to avoid segfault when trying to use torch tensor in extension
  # separate pybind11 from libtriton.so
  # import like this: from triton._C import _metal_driver
  add_library(_metal_driver MODULE ${CMAKE_CURRENT_SOURCE_DIR}/backend/driver.mm)
  target_include_directories(_metal_driver PRIVATE ${TORCH_INCLUDE_DIRS})
  set_target_properties(_metal_driver PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH "${TORCH_LIB_DIR}"
    INSTALL_RPATH "${TORCH_LIB_DIR}"
  )
  target_link_libraries(_metal_driver PRIVATE
    Python3::Module
    ${TORCH_PATH}/lib/libtorch_python.dylib
    ${TORCH_PATH}/lib/libtorch.dylib
    ${TORCH_PATH}/lib/libtorch_cpu.dylib
    ${TORCH_PATH}/lib/libc10.dylib
  )
endif()