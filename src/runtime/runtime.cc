#include "infinirt.h"

// Device
__C infinirtStatus_t setDevice(DeviceType device, int deviceId);

// Stream
__C infinirtStatus_t createStream(stream_t* pStream, DeviceType device, int deviceId);
__C infinirtStatus_t destoryStream(DeviceType device, int deviceId, stream_t stream);

// Event
__C infinirtStatus_t createEvent(event_t* pEvent, DeviceType device, int deviceId, stream_t stream);
__C infinirtStatus_t destoryEvent(event_t event, DeviceType device, int deviceId);
__C infinirtStatus_t waitEvent(event_t event, DeviceType device, int deviceId, stream_t stream);

// Memory
__C infinirtStatus_t *deviceMalloc(DeviceType device, int deviceId, size_t size);
__C infinirtStatus_t *deviceMallocAsync(DeviceType device, int deviceId, size_t size, stream_t stream);
__C infinirtStatus_t deviceFree(void *ptr, DeviceType device, int deviceId);
__C infinirtStatus_t deviceFreeAsync(void *ptr, DeviceType device, int deviceId, stream_t stream);
__C infinirtStatus_t memcpyH2DAsync(void *dst, const void *src, size_t size, DeviceType device, int deviceId, stream_t stream);
__C infinirtStatus_t memcpyD2H(void *dst, const void *src, size_t size, DeviceType device, int deviceId);

