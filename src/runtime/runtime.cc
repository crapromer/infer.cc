#include "runtime.h"
#include "ascend/infinirt_ascend.h"
#include "cuda/infinirt_cuda.h"
#include "teco/infinirt_teco.h"
#include <cstdlib>
#include <string.h>

__C __export infinirtStatus_t infinirtInit(DeviceType device){
    switch (device){
        case DEVICE_CPU:
            return INFINIRT_STATUS_SUCCESS;
        case DEVICE_NVIDIA:
            return INFINIRT_STATUS_SUCCESS;
        case DEVICE_ASCEND:
            return initAscend();
        case DEVICE_TECO:
            return INFINIRT_STATUS_SUCCESS;
        default:
            return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Device
__C infinirtStatus_t infinirtDeviceSynchronize(DeviceType device, uint32_t deviceId){
    switch (device)
    {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaDevice(deviceId);
    case DEVICE_ASCEND:
        return synchronizeAscendDevice(deviceId);
    case DEVICE_TECO:
        return synchronizeTecoDevice(deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Stream
__C infinirtStatus_t infinirtStreamCreate(infinirtStream_t *pStream, DeviceType device, uint32_t deviceId)
{
    switch (device)
    {
    case DEVICE_CPU:
    {
        auto stream = new infinirtStream();
        stream->device = DEVICE_CPU;
        stream->device_id = 0;
        stream->stream = nullptr;
        (*pStream) = stream;
        return INFINIRT_STATUS_SUCCESS;
    }
        
    case DEVICE_NVIDIA:
        return createCudaStream(pStream, deviceId);
    case DEVICE_ASCEND:
        return createAscendStream(pStream, deviceId);
    case DEVICE_TECO:
        return createTecoStream(pStream, deviceId);
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
    case DEVICE_CPU:
        delete stream;
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return destoryCudaStream(stream);
    case DEVICE_ASCEND:
        return destoryAscendStream(stream);
    case DEVICE_TECO:
        return destoryTecoStream(stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtStreamSynchronize(infinirtStream_t stream){
    if (stream == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (stream->device)
    {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaStream(stream);
    case DEVICE_ASCEND:
        return synchronizeAscendStream(stream);
    case DEVICE_TECO:
        return synchronizeTecoStream(stream);
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

__C infinirtStatus_t infinirtGetStreamDeviceInfo(DeviceType *deviceType,
                                                 uint32_t *deviceId,
                                                 infinirtStream_t stream) {
    if (stream == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (deviceType != nullptr) {
        *deviceType = stream->device;
    }
    if (deviceId != nullptr) {
        *deviceId = stream->device_id;
    }
    return INFINIRT_STATUS_SUCCESS;
}

// Event
__C infinirtStatus_t infinirtEventCreate(infinirtEvent_t *pEvent, DeviceType device, uint32_t deviceId)
{
    switch (device)
    {
    case DEVICE_CPU:
    {
        auto event = new infinirtEvent();
        event->device = DEVICE_CPU;
        event->device_id = 0;
        event->event = nullptr;
        (*pEvent) = event;
        return INFINIRT_STATUS_SUCCESS;
    }
        
    case DEVICE_NVIDIA:
        return createCudaEvent(pEvent, deviceId);
    case DEVICE_ASCEND:
        return createAscendEvent(pEvent, deviceId);
    case DEVICE_TECO:
        return createTecoEvent(pEvent,deviceId);
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
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return recordCudaEvent(event, stream);
    case DEVICE_ASCEND:
        return recordAscendEvent(event, stream);
    case DEVICE_TECO:
        return recordTecoEvent(event,stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventQuery(infinirtEvent_t event) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (event->device) {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return queryCudaEvent(event);
    case DEVICE_ASCEND:
        return queryAscendEvent(event);
    case DEVICE_TECO:
        return queryTecoEvent(event);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventSynchronize(infinirtEvent_t event) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (event->device) {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaEvent(event);
    case DEVICE_ASCEND:
        return synchronizeAscendEvent(event);
    case DEVICE_TECO:
        return synchronizeTecoEvent(event);
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
    case DEVICE_CPU:
        delete event;
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return destoryCudaEvent(event);
    case DEVICE_ASCEND:
        return destoryAscendEvent(event);
    case DEVICE_TECO:
        return destoryTecoEvent(event);
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
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return waitCudaEvent(event, stream);
    case DEVICE_ASCEND:
        return waitAscendEvent(event, stream);
    case DEVICE_TECO:
        return waitTecoEvent(event,stream);
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
    case DEVICE_ASCEND:
        return mallocAscend(pMemory, deviceId, size);
    case DEVICE_TECO:
        return mallocTeco(pMemory,deviceId,size);
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
    case DEVICE_ASCEND:
        return mallocAscendAsync(pMemory, deviceId, size, stream);
    case DEVICE_TECO:
        return mallocAscendAsync(pMemory, deviceId, size, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtMallocHost(void **pMemory,
                                                 DeviceType device,
                                                 uint32_t deviceId,
                                                 size_t size) {
    switch (device) {
    case DEVICE_CPU:
        return infinirtMalloc(pMemory, device, deviceId, size);
    case DEVICE_NVIDIA:
        return mallocHostCuda(pMemory, deviceId, size);
    case DEVICE_ASCEND:
        return mallocHostAscend(pMemory, deviceId, size);
    case DEVICE_TECO:
        return mallocHostTeco(pMemory,deviceId,size);
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
    case DEVICE_ASCEND:
        return freeAscend(ptr, deviceId);
    case DEVICE_TECO:
        return freeTeco(ptr,deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFreeAsync(void *ptr, DeviceType device,
                                       uint32_t deviceId,
                                       infinirtStream_t stream) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    if (stream == nullptr) {
        return infinirtFree(ptr, device, deviceId);
    } else if (device != stream->device || deviceId != stream->device_id) {
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    }
    switch (device) {
    case DEVICE_CPU:
        return infinirtFree(ptr, device, deviceId);
    case DEVICE_NVIDIA:
        return freeCudaAsync(ptr, deviceId, stream);
    case DEVICE_ASCEND:
        return freeAscendAsync(ptr, deviceId, stream);
    case DEVICE_TECO:
        return freeTecoAsync(ptr,deviceId,stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtFreeHost(void *ptr, DeviceType device,
                                               uint32_t deviceId) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (device) {
    case DEVICE_CPU:
        return infinirtFree(ptr, device, deviceId);
    case DEVICE_NVIDIA:
        return freeHostCuda(ptr, deviceId);
    case DEVICE_ASCEND:
        return freeHostAscend(ptr, deviceId);
    case DEVICE_TECO:
        return freeHostTeco(ptr,deviceId);
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
    case DEVICE_ASCEND:
        return memcpyHost2Ascend(dst, deviceId, src, size);
    case DEVICE_TECO:
        return memcpyHost2Teco(dst,deviceId,src,size);
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
    case DEVICE_ASCEND:
        return memcpyHost2AscendAsync(dst, deviceId, src, size, stream);
    case DEVICE_TECO:
        return memcpyHost2TecoAsync(dst,deviceId,src,size,stream);
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
    case DEVICE_CPU:
        memcpy(dst, src, size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return memcpyCuda2Host(dst, src, deviceId, size);
    case DEVICE_ASCEND:
        return memcpyAscend2Host(dst, src, deviceId, size);
    case DEVICE_TECO:
        return memcpyTeco2Host(dst,src,deviceId,size);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtMemcpy(void *dst, const void *src,
                                             DeviceType device,
                                             uint32_t deviceId, size_t size) {
    if (size == 0)
        return INFINIRT_STATUS_SUCCESS;
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (device) {
    case DEVICE_CPU:
        memcpy(dst, src, size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return memcpyCuda(dst, src, deviceId, size);
    case DEVICE_ASCEND:
        return memcpyAscend(dst, src, deviceId, size);
    case DEVICE_TECO:
        return memcpyTeco(dst,src,deviceId,size);
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
    case DEVICE_CPU:
        return infinirtMemcpy(dst, src, device, deviceId, size);
    case DEVICE_NVIDIA:
        return memcpyCudaAsync(dst, src, deviceId, size, stream);
    case DEVICE_ASCEND:
        return memcpyAscendAsync(dst, src, deviceId, size, stream);
    case DEVICE_TECO:
        return memcpyTecoAsync(dst, src, deviceId, size, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
