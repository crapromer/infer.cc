#ifndef RUNTIME_H
#define RUNTIME_H
#include "infini_infer.h"

// Device
void setDevice(Device device, int deviceId);

// Stream
typedef void *stream_t;
#define DEFAULT_STREAM ((stream_t)0)
stream_t createStream(Device device, int deviceId);
void destoryStream(Device device, int deviceId, stream_t stream);

// Event
typedef void *event_t;
event_t createEvent(Device device, int deviceId, stream_t stream);
void destoryEvent(event_t event, Device device, int deviceId);
void waitEvent(event_t event, Device device, int deviceId, stream_t stream);

// Memory
void *deviceMalloc(Device device, int deviceId, size_t size);
void *deviceMallocAsync(Device device, int deviceId, size_t size, stream_t stream);
void deviceFree(void *ptr, Device device, int deviceId);
void deviceFreeAsync(void *ptr, Device device, int deviceId, stream_t stream);
void memcpyH2DAsync(void *dst, const void *src, size_t size, Device device, int deviceId, stream_t stream);
void memcpyD2H(void *dst, const void *src, size_t size, Device device, int deviceId);

#endif
