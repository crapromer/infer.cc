#ifndef INFINIRT_CUDA_H
#define INFINIRT_CUDA_H
#include "../runtime.h"

#ifdef ENABLE_NV_GPU
#define IMPL_WITH_CUDA ;
#else
#define IMPL_WITH_CUDA {  return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t synchronizeCudaDevice(uint32_t deviceId) IMPL_WITH_CUDA

infinirtStatus_t createCudaStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_CUDA
infinirtStatus_t destoryCudaStream(infinirtStream_t stream) IMPL_WITH_CUDA

infinirtStatus_t createCudaEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_CUDA
infinirtStatus_t destoryCudaEvent(infinirtEvent_t event) IMPL_WITH_CUDA
infinirtStatus_t waitCudaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t recordCudaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t queryCudaEvent(infinirtEvent_t event) IMPL_WITH_CUDA
infinirtStatus_t synchronizeCudaEvent(infinirtEvent_t event) IMPL_WITH_CUDA

infinirtStatus_t mallocCuda(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_CUDA
infinirtStatus_t mallocCudaAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t freeCuda(void *ptr, uint32_t deviceId) IMPL_WITH_CUDA
infinirtStatus_t freeCudaAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t memcpyHost2Cuda(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_CUDA
infinirtStatus_t memcpyHost2CudaAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t memcpyCuda2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_CUDA
infinirtStatus_t memcpyCudaAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_CUDA
#endif
