#ifndef INFINIRT_CUDA_H
#define INFINIRT_CUDA_H
#include "../runtime.h"

#ifdef ENABLE_NV_GPU
#define IMPL_WITH_CUDA ;
#else
#define IMPL_WITH_CUDA { return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t createCudaStream(infiniStream_t *pStream, uint32_t deviceId) IMPL_WITH_CUDA
infinirtStatus_t destoryCudaStream(infiniStream_t stream) IMPL_WITH_CUDA

infinirtStatus_t createCudaEvent(infiniEvent_t *pEvent, infiniStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t destoryCudaEvent(infiniEvent_t event) IMPL_WITH_CUDA
infinirtStatus_t waitCudaEvent(infiniEvent_t event, infiniStream_t stream) IMPL_WITH_CUDA

infinirtStatus_t mallocCuda(infiniMemory_t *pMemory, uint32_t deviceId, size_t size) IMPL_WITH_CUDA
infinirtStatus_t mallocCudaAsync(infiniMemory_t *pMemory, uint32_t deviceId, size_t size, infiniStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t freeCuda(infiniMemory_t ptr) IMPL_WITH_CUDA
infinirtStatus_t freeCudaAsync(infiniMemory_t ptr, infiniStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t memcpyHost2CudaAsync(infiniMemory_t dst, const void *src, size_t size, infiniStream_t stream) IMPL_WITH_CUDA
infinirtStatus_t memcpyCuda2Host(void *dst, const infiniMemory_t src, size_t size) IMPL_WITH_CUDA


#endif
