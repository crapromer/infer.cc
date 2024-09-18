#ifndef INFINI_RUNTIME_H
#define INFINI_RUNTIME_H

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#else
#define __C
#endif

typedef enum
{
    DEVICE_CPU,
    DEVICE_NVIDIA,
    DEVICE_CAMBRICON,
} DeviceType;

// Device
__C __export void setDevice(DeviceType device, int deviceId);

// Stream
typedef void *stream_t;
#define DEFAULT_STREAM ((stream_t)0)
__C __export stream_t createStream(DeviceType device, int deviceId);
__C __export void destoryStream(DeviceType device, int deviceId, stream_t stream);

// Event
typedef void *event_t;
__C __export event_t createEvent(DeviceType device, int deviceId, stream_t stream);
__C __export void destoryEvent(event_t event, DeviceType device, int deviceId);
__C __export void waitEvent(event_t event, DeviceType device, int deviceId, stream_t stream);

// Memory
__C __export void *deviceMalloc(DeviceType device, int deviceId, size_t size);
__C __export void *deviceMallocAsync(DeviceType device, int deviceId, size_t size, stream_t stream);
__C __export void deviceFree(void *ptr, DeviceType device, int deviceId);
__C __export void deviceFreeAsync(void *ptr, DeviceType device, int deviceId, stream_t stream);
__C __export void memcpyH2DAsync(void *dst, const void *src, size_t size, DeviceType device, int deviceId, stream_t stream);
__C __export void memcpyD2H(void *dst, const void *src, size_t size, DeviceType device, int deviceId);

#endif
