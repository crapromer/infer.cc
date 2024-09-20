#include "runtime.h"
#include "cuda/infinirt_cuda.h"

// Stream
__C infinirtStatus_t infinirtStreamCreate(infinirtStream_t *pStream, DeviceType device, uint32_t deviceId)
{
    switch (device)
    {
    case DEVICE_NVIDIA:
        return createCudaStream(pStream, deviceId);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtStreamDestroy(infinirtStream_t stream)
{
    if (stream == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (stream->device)
    {
    case DEVICE_NVIDIA:
        return destoryCudaStream(stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Event
__C infinirtStatus_t infinirtEventCreate(infinirtEvent_t *pEvent, DeviceType device, uint32_t deviceId)
{
    switch (device)
    {
    case DEVICE_NVIDIA:
        return createCudaEvent(pEvent, deviceId);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtEventDestroy(infinirtEvent_t event)
{
    if (event == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (event->device)
    {
    case DEVICE_NVIDIA:
        return destoryCudaEvent(event);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtStreamWaitEvent(infinirtEvent_t event, infinirtStream_t stream)
{
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr && event->device != stream->device)
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (event->device)
    {
    case DEVICE_NVIDIA:
        return waitCudaEvent(event, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Memory
__C infinirtStatus_t infinirtMalloc(infinirtMemory_t *pMemory, DeviceType device, uint32_t deviceId, size_t size)
{
    switch (device)
    {
    case DEVICE_NVIDIA:
        return mallocCuda(pMemory, deviceId, size);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMallocAsync(infinirtMemory_t *pMemory, DeviceType device, uint32_t deviceId, size_t size, infinirtStream_t stream)
{
    if (stream != nullptr && device != stream->device)
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (device)
    {
    case DEVICE_NVIDIA:
        return mallocCudaAsync(pMemory, deviceId, size, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFree(infinirtMemory_t ptr)
{
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (ptr->device)
    {
    case DEVICE_NVIDIA:
        return freeCuda(ptr);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFreeAsync(infinirtMemory_t ptr, infinirtStream_t stream)
{
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    if (stream != nullptr && ptr->device != stream->device)
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (ptr->device)
    {
    case DEVICE_NVIDIA:
        return freeCudaAsync(ptr, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyH2DAsync(infinirtMemory_t dst, const void *src, size_t size, infinirtStream_t stream)
{
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr && dst->device != stream->device)
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    if (size > dst->size && dst->device != DEVICE_CPU)
        return INFINIRT_STATUS_ILLEGAL_MEMORY_ACCESS;

    switch (dst->device)
    {
    case DEVICE_NVIDIA:
        return memcpyHost2CudaAsync(dst, src, size, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyD2H(void *dst, const infinirtMemory_t src, size_t size)
{
    if (src == nullptr || dst == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (size > src->size)
        return INFINIRT_STATUS_ILLEGAL_MEMORY_ACCESS;

    switch (src->device)
    {
    case DEVICE_NVIDIA:
        return memcpyCuda2Host(dst, src, size);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
