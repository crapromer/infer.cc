#ifndef INFINIRT_CUDA_H
#define INFINIRT_CUDA_H
#include "../runtime.h"

#ifdef ENABLE_NV_GPU
#define IMPL_WITH_CUDA ;
#else
#define IMPL_WITH_CUDA                               \
    {                                                \
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; \
    }
#endif

infinirtStatus_t createCudaStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_CUDA
    infinirtStatus_t destoryCudaStream(infinirtStream_t stream) IMPL_WITH_CUDA

    infinirtStatus_t createCudaEvent(infinirtEvent_t *pEvent, infinirtStream_t stream) IMPL_WITH_CUDA
    infinirtStatus_t destoryCudaEvent(infinirtEvent_t event) IMPL_WITH_CUDA
    infinirtStatus_t waitCudaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_CUDA

    infinirtStatus_t mallocCuda(infinirtMemory_t *pMemory, uint32_t deviceId, size_t size) IMPL_WITH_CUDA
    infinirtStatus_t mallocCudaAsync(infinirtMemory_t *pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_CUDA
    infinirtStatus_t freeCuda(infinirtMemory_t ptr) IMPL_WITH_CUDA
    infinirtStatus_t freeCudaAsync(infinirtMemory_t ptr, infinirtStream_t stream) IMPL_WITH_CUDA
    infinirtStatus_t memcpyHost2CudaAsync(infinirtMemory_t dst, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_CUDA
    infinirtStatus_t memcpyCuda2Host(void *dst, const infinirtMemory_t src, size_t size) IMPL_WITH_CUDA

#endif
