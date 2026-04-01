#include "torch/csrc/autograd/python_variable.h"
#include "torch/mps.h"
#include <ATen/native/mps/OperationUtils.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace {

struct MetalKernel {
  id<MTLLibrary> lib;
  id<MTLComputePipelineState> pipeline;
  NSUInteger max_threads;
};

std::mutex cache_mutex;
std::unordered_map<int, std::shared_ptr<MetalKernel>> id_to_kernel_map;
std::unordered_map<std::string, int> key_to_id_map;
int next_id = 1;

id<MTLLibrary> create_lib(id<MTLDevice> dev, const std::string &name,
                          const std::string &metallib_bytes) {
  // write metallib bytes to temp file and create MTLLibrary from that file
  NSData *data = [NSData dataWithBytes:metallib_bytes.data()
                                length:metallib_bytes.size()];
  NSString *tmp_dir = NSTemporaryDirectory();
  NSString *file_name =
      [[NSUUID UUID].UUIDString stringByAppendingPathExtension:@"metallib"];
  NSString *file_path = [tmp_dir stringByAppendingPathComponent:file_name];
  NSURL *file_url = [NSURL fileURLWithPath:file_path];
  NSError *error = nil;
  if (![data writeToURL:file_url options:NSDataWritingAtomic error:&error]) {
    if (error != nil) {
      throw std::runtime_error(
          "failed to write metallib for " + name + ": " +
          std::string([[error localizedDescription] UTF8String]));
    }
    throw std::runtime_error("failed to write metallib for " + name);
  }

  error = nil;
  id<MTLLibrary> lib = [dev newLibraryWithURL:file_url error:&error];
  [[NSFileManager defaultManager] removeItemAtURL:file_url error:nil];
  if (lib != nil) {
    return lib;
  }
  if (error != nil) {
    throw std::runtime_error(
        "failed to create lib for " + name + ": " +
        std::string([[error localizedDescription] UTF8String]));
  }
  throw std::runtime_error("failed to create lib for " + name);
}

// create cache key for this kernel
std::string make_kernel_key(const std::string &name,
                            const py::bytes &metallib) {
  std::string bytes = metallib;
  std::string key;
  key.reserve(name.size() + 1 + bytes.size());
  key.append(name);
  key.push_back('\0');
  key.append(bytes);
  return key;
}

// create MetalKernel for this kernel to store in the global cache
std::shared_ptr<MetalKernel> create_binary(const std::string &name,
                                           const py::bytes &metallib) {
  std::string metallib_bytes = metallib;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (dev == nil) {
    throw std::runtime_error("failed to create metal device");
  }

  NSError *error = nil;
  id<MTLLibrary> lib = create_lib(dev, name, metallib_bytes);

  NSString *function_name = [NSString stringWithUTF8String:name.c_str()];
  id<MTLFunction> func = [lib newFunctionWithName:function_name];
  if (func == nil) {
    throw std::runtime_error(name + " not found");
  }

  id<MTLComputePipelineState> pipeline =
      [dev newComputePipelineStateWithFunction:func error:&error];
  if (pipeline == nil) {
    throw std::runtime_error([[error localizedDescription] UTF8String]);
  }

  return std::make_shared<MetalKernel>(
      MetalKernel{lib, pipeline, [pipeline maxTotalThreadsPerThreadgroup]});
}

} // namespace

PYBIND11_MODULE(_metal_buffer, m) {
  m.def("get_mtl_buffer", [](py::object obj) -> uintptr_t {
    at::Tensor t = THPVariable_Unpack(obj.ptr());
    id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(t);
    return reinterpret_cast<uintptr_t>(buf);
  });

  m.def("get_command_buffer", []() -> uintptr_t {
    id<MTLCommandBuffer> buf = torch::mps::get_command_buffer();
    return reinterpret_cast<uintptr_t>(buf);
  });

  m.def("load_binary", [](const std::string &name, py::bytes metallib) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    std::string key = make_kernel_key(name, metallib);
    int id = 0;
    std::shared_ptr<MetalKernel> binary;

    auto it = key_to_id_map.find(key);
    if (it != key_to_id_map.end()) {
      // already cached
      id = it->second;
      binary = id_to_kernel_map.at(id);
    } else {
      // create and cache kernel representation
      id = next_id++;
      binary = create_binary(name, metallib);
      key_to_id_map.emplace(std::move(key), id);
      id_to_kernel_map.emplace(id, binary);
    }

    return py::make_tuple(id, reinterpret_cast<uintptr_t>(binary->pipeline), 0,
                          0, static_cast<int>(binary->max_threads));
  });

  m.def("unload_module", [](int id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto id_it = id_to_kernel_map.find(id);
    if (id_it == id_to_kernel_map.end()) {
      return;
    }

    // search for id in cache and remove kernel
    for (auto key_it = key_to_id_map.begin(); key_it != key_to_id_map.end();
         ++key_it) {
      if (key_it->second == id) {
        key_to_id_map.erase(key_it);
        break;
      }
    }
    id_to_kernel_map.erase(id_it);
  });
}
