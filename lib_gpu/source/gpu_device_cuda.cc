#include "precomp.h"

#include "core_interface.h"
#include "gpu_device_cuda.h"

namespace lib_gpu {
GpuDeviceCuda::GpuDeviceCuda(int dev_id) : GpuDevice(dev_id) {
  int nr_of_gpus = 0;
  if (!CheckCudaError(cuDeviceGetCount(&nr_of_gpus))) return;
  device_count_ = nr_of_gpus;
  if (device_count_ == 0) return;

  cuda_context_ = std::make_shared<CudaDeviceContext>(dev_id);
}

void GpuDeviceCuda::PushContextOnThread() {
  cuCtxPushCurrent(cuda_context_->context_);
}

void GpuDeviceCuda::SynchronizeDevice(int stream) {
  CUresult error;
  if (stream == -1)
    error = cuCtxSynchronize();
  else
    error = cuStreamSynchronize(cuda_context_->streams_[stream]);
  CheckCudaError(error);
}
void GpuDeviceCuda::DeallocateMemory(void *dev_ptr) {
  CUresult error = cuMemFree(reinterpret_cast<CUdeviceptr>(dev_ptr));
  CheckCudaError(error);
}
void GpuDeviceCuda::DeallocateHostMemory(void *host_ptr) {
  CUresult error = cuMemFreeHost(host_ptr);
  CheckCudaError(error);
}
void GpuDeviceCuda::AllocateManagedMemory(void **dev_ptr, size_t size) {
  CUresult error = cuMemAllocManaged(reinterpret_cast<CUdeviceptr *>(dev_ptr),
                                     size, CU_MEM_ATTACH_GLOBAL);
  CheckCudaError(error);
}
void GpuDeviceCuda::CopyToDevice(void *host_ptr, void *dev_ptr, size_t size,
                                 int stream) {
  cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dev_ptr), host_ptr, size,
                    cuda_context_->streams_[stream]);
}
void GpuDeviceCuda::CopyToHost(void *host_ptr, void *dev_ptr, size_t size,
                               int stream) {
  cuMemcpyDtoHAsync(host_ptr, reinterpret_cast<CUdeviceptr>(dev_ptr), size,
                    cuda_context_->streams_[stream]);
}
void GpuDeviceCuda::AllocateMemory(void **dev_ptr, size_t size) {
  CUresult error = cuMemAlloc(reinterpret_cast<CUdeviceptr *>(dev_ptr), size);
  CheckCudaError(error);
}
void GpuDeviceCuda::AllocateHostMemory(void **dev_ptr, size_t size) {
  CUresult error = cuMemAllocHost(dev_ptr, size);
  CheckCudaError(error);
}

bool GpuDeviceCuda::CheckCudaError(CUresult error) {
  auto &core_interface = lib_core::CoreInterface::GetInstance();
  string error_string = "";

  switch (error) {
    case CUresult::CUDA_ERROR_ALREADY_ACQUIRED:
      error_string = "CUDA_ERROR_ALREADY_ACQUIRED";
      break;
    case CUresult::CUDA_ERROR_ALREADY_MAPPED:
      error_string = "CUDA_ERROR_ALREADY_MAPPED";
      break;
    case CUresult::CUDA_ERROR_ARRAY_IS_MAPPED:
      error_string = "CUDA_ERROR_ARRAY_IS_MAPPED";
      break;
    case CUresult::CUDA_ERROR_ASSERT:
      error_string = "CUDA_ERROR_ASSERT";
      break;
    case CUresult::CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
      error_string = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
      break;
    case CUresult::CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
      error_string = "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
      break;
    case CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED:
      error_string = "CUDA_ERROR_CONTEXT_IS_DESTROYED";
      break;
    case CUresult::CUDA_ERROR_DEINITIALIZED:
      error_string = "CUDA_ERROR_DEINITIALIZED";
      break;
    case CUresult::CUDA_ERROR_ECC_UNCORRECTABLE:
      error_string = "CUDA_ERROR_ECC_UNCORRECTABLE";
      break;
    case CUresult::CUDA_ERROR_FILE_NOT_FOUND:
      error_string = "CUDA_ERROR_FILE_NOT_FOUND";
      break;
    case CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR:
      error_string = "CUDA_ERROR_HARDWARE_STACK_ERROR";
      break;
    case CUresult::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      error_string = "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
      break;
    case CUresult::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
      error_string = "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
      break;
    case CUresult::CUDA_ERROR_ILLEGAL_ADDRESS:
      error_string = "CUDA_ERROR_ILLEGAL_ADDRESS";
      break;
    case CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION:
      error_string = "CUDA_ERROR_ILLEGAL_INSTRUCTION";
      break;
    case CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE:
      error_string = "CUDA_ERROR_INVALID_ADDRESS_SPACE";
      break;
    case CUresult::CUDA_ERROR_INVALID_CONTEXT:
      error_string = "CUDA_ERROR_INVALID_CONTEXT";
      break;
    case CUresult::CUDA_ERROR_INVALID_DEVICE:
      error_string = "CUDA_ERROR_INVALID_DEVICE";
      break;
    case CUresult::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
      error_string = "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
      break;
    case CUresult::CUDA_ERROR_INVALID_HANDLE:
      error_string = "CUDA_ERROR_INVALID_HANDLE";
      break;
    case CUresult::CUDA_ERROR_INVALID_IMAGE:
      error_string = "CUDA_ERROR_INVALID_IMAGE";
      break;
    case CUresult::CUDA_ERROR_INVALID_PC:
      error_string = "CUDA_ERROR_INVALID_PC";
      break;
    case CUresult::CUDA_ERROR_INVALID_PTX:
      error_string = "CUDA_ERROR_INVALID_PTX";
      break;
    case CUresult::CUDA_ERROR_INVALID_SOURCE:
      error_string = "CUDA_ERROR_INVALID_SOURCE";
      break;
    case CUresult::CUDA_ERROR_INVALID_VALUE:
      error_string = "CUDA_ERROR_INVALID_VALUE";
      break;
    case CUresult::CUDA_ERROR_LAUNCH_FAILED:
      error_string = "CUDA_ERROR_LAUNCH_FAILED";
      break;
    case CUresult::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
      error_string = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
      break;
    case CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      error_string = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
      break;
    case CUresult::CUDA_ERROR_LAUNCH_TIMEOUT:
      error_string = "CUDA_ERROR_LAUNCH_TIMEOUT";
      break;
    case CUresult::CUDA_ERROR_MAP_FAILED:
      error_string = "CUDA_ERROR_MAP_FAILED";
      break;
    case CUresult::CUDA_ERROR_MISALIGNED_ADDRESS:
      error_string = "CUDA_ERROR_MISALIGNED_ADDRESS";
      break;
    case CUresult::CUDA_ERROR_NOT_FOUND:
      error_string = "CUDA_ERROR_NOT_FOUND";
      break;
    case CUresult::CUDA_ERROR_NOT_INITIALIZED:
      error_string = "CUDA_ERROR_NOT_INITIALIZED";
      break;
    case CUresult::CUDA_ERROR_NOT_MAPPED:
      error_string = "CUDA_ERROR_NOT_MAPPED";
      break;
    case CUresult::CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
      error_string = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
      break;
    case CUresult::CUDA_ERROR_NOT_MAPPED_AS_POINTER:
      error_string = "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
      break;
    case CUresult::CUDA_ERROR_NOT_PERMITTED:
      error_string = "CUDA_ERROR_NOT_PERMITTED";
      break;
    case CUresult::CUDA_ERROR_NOT_READY:
      error_string = "CUDA_ERROR_NOT_READY";
      break;
    case CUresult::CUDA_ERROR_NOT_SUPPORTED:
      error_string = "CUDA_ERROR_NOT_SUPPORTED";
      break;
    case CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU:
      error_string = "CUDA_ERROR_NO_BINARY_FOR_GPU";
      break;
    case CUresult::CUDA_ERROR_NO_DEVICE:
      error_string = "CUDA_ERROR_NO_DEVICE";
      break;
    case CUresult::CUDA_ERROR_NVLINK_UNCORRECTABLE:
      error_string = "CUDA_ERROR_NVLINK_UNCORRECTABLE";
      break;
    case CUresult::CUDA_ERROR_OPERATING_SYSTEM:
      error_string = "CUDA_ERROR_OPERATING_SYSTEM";
      break;
    case CUresult::CUDA_ERROR_OUT_OF_MEMORY:
      error_string = "CUDA_ERROR_OUT_OF_MEMORY";
      break;
    case CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      error_string = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
      break;
    case CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      error_string = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
      break;
    case CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
      error_string = "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
      break;
    case CUresult::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
      error_string = "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
      break;
    case CUresult::CUDA_ERROR_PROFILER_ALREADY_STARTED:
      error_string = "CUDA_ERROR_PROFILER_ALREADY_STARTED";
      break;
    case CUresult::CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      error_string = "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
      break;
    case CUresult::CUDA_ERROR_PROFILER_DISABLED:
      error_string = "CUDA_ERROR_PROFILER_DISABLED";
      break;
    case CUresult::CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      error_string = "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
      break;
    case CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      error_string = "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
      break;
    case CUresult::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      error_string = "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
      break;
    case CUresult::CUDA_ERROR_TOO_MANY_PEERS:
      error_string = "CUDA_ERROR_TOO_MANY_PEERS";
      break;
    case CUresult::CUDA_ERROR_UNKNOWN:
      error_string = "CUDA_ERROR_UNKNOWN";
      break;
    case CUresult::CUDA_ERROR_UNMAP_FAILED:
      error_string = "CUDA_ERROR_UNMAP_FAILED";
      break;
    case CUresult::CUDA_ERROR_UNSUPPORTED_LIMIT:
      error_string = "CUDA_ERROR_UNSUPPORTED_LIMIT";
      break;
    case CUresult::CUDA_SUCCESS:
      return true;
    default:
      error_string = "Unknown";
      break;
  }

  core_interface.ThrowException("Cuda error found: " + error_string);
  return false;
}
GpuDeviceCuda::CudaDeviceContext::CudaDeviceContext(int dev_id) {
  CUdevice device;
  cuDeviceGet(&device, dev_id);
  cuCtxCreate(&context_, 0, device);

  for (int i = 0; i < 3; ++i) {
    streams_.emplace_back(CUstream());
    cuStreamCreate(&streams_.back(), CU_STREAM_NON_BLOCKING);
  }

  cuCtxPushCurrent(context_);
}
GpuDeviceCuda::CudaDeviceContext::~CudaDeviceContext() {
  for (int i = 0; i < streams_.size(); ++i) cuStreamDestroy(streams_[i]);
  cuCtxDestroy(context_);
}
}