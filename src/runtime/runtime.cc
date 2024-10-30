#include "runtime.h"
#include "cuda/infinirt_cuda.h"
#include <cstdlib>
#include <string.h>

// Device
__C infinirtStatus_t infinirtDeviceSynchronize(DeviceType device, uint32_t deviceId){
    switch (device)
    {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaDevice(deviceId);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

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

__C infinirtStatus_t infinirtGetRawStream(void **ptr, infinirtStream_t stream) {
    if (stream == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    *ptr = stream->stream;
    return INFINIRT_STATUS_SUCCESS;
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
__C infinirtStatus_t infinirtEventRecord(infinirtEvent_t event,
                                         infinirtStream_t stream) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr && event->device != stream->device)
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (event->device) {
    case DEVICE_NVIDIA:
        return recordCudaEvent(event, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventQuery(infinirtEvent_t event) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (event->device) {
    case DEVICE_NVIDIA:
        return queryCudaEvent(event);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventSynchronize(infinirtEvent_t event) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (event->device) {
    case DEVICE_NVIDIA:
        return synchronizeCudaEvent(event);

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
    if (stream != nullptr && (event->device != stream->device ||
                              stream->device_id != event->device_id))
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
__C infinirtStatus_t infinirtMalloc(void **pMemory, DeviceType device,
                                    uint32_t deviceId, size_t size) {
    switch (device)
    {
    case DEVICE_CPU:
        *pMemory = std::malloc(size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return mallocCuda(pMemory, deviceId, size);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMallocAsync(void **pMemory, DeviceType device,
                                         uint32_t deviceId, size_t size,
                                         infinirtStream_t stream) {
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (device)
    {
    case DEVICE_CPU:
        return infinirtMalloc(pMemory, device, deviceId, size);
    case DEVICE_NVIDIA:
        return mallocCudaAsync(pMemory, deviceId, size, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFree(void *ptr, DeviceType device,
                                  uint32_t deviceId) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (device) {
    case DEVICE_CPU:
        std::free(ptr);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return freeCuda(ptr, deviceId);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFreeAsync(void *ptr, DeviceType device,
                                       uint32_t deviceId,
                                       infinirtStream_t stream) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (device) {
    case DEVICE_CPU:
        return infinirtFree(ptr, device, deviceId);
    case DEVICE_NVIDIA:
        return freeCudaAsync(ptr, deviceId, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyH2D(void *dst, DeviceType device,
                                            uint32_t deviceId, const void *src,
                                            size_t size) {
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;

    switch (device) {
    case DEVICE_CPU:
        memcpy(dst, src, size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return memcpyHost2Cuda(dst, deviceId, src, size);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyH2DAsync(void *dst, DeviceType device,
                                            uint32_t deviceId, const void *src,
                                            size_t size,
                                            infinirtStream_t stream) {
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;

    switch (device) {
    case DEVICE_CPU:
        return infinirtMemcpyH2D(dst, device, deviceId, src, size);
    case DEVICE_NVIDIA:
        return memcpyHost2CudaAsync(dst, deviceId, src, size, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyD2H(void *dst, const void *src,
                                       DeviceType device, uint32_t deviceId,
                                       size_t size) {
    if (src == nullptr || dst == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;

    switch (device) {
    case DEVICE_NVIDIA:
        return memcpyCuda2Host(dst, src, deviceId, size);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtMemcpyAsync(void *dst, const void *src,
                                                  DeviceType device,
                                                  uint32_t deviceId,
                                                  size_t size,
                                                  infinirtStream_t stream) {
    if (size == 0)
        return INFINIRT_STATUS_SUCCESS;
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;

    switch (device) {
    case DEVICE_NVIDIA:
        return memcpyCudaAsync(dst, src, deviceId, size, stream);

    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
