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

typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_EXECUTION_FAILED = 1,
} infinirtStatus_t;

// Device
__C __export infinirtStatus_t setDevice(DeviceType device, int deviceId);

// Stream
typedef void *stream_t;
#define DEFAULT_STREAM ((stream_t)0)
__C __export infinirtStatus_t createStream(stream_t* pStream, DeviceType device, int deviceId);
__C __export infinirtStatus_t destoryStream(DeviceType device, int deviceId, stream_t stream);

// Event
typedef void *event_t;
__C __export infinirtStatus_t createEvent(event_t* pEvent, DeviceType device, int deviceId, stream_t stream);
__C __export infinirtStatus_t destoryEvent(event_t event, DeviceType device, int deviceId);
__C __export infinirtStatus_t waitEvent(event_t event, DeviceType device, int deviceId, stream_t stream);

// Memory
__C __export infinirtStatus_t *deviceMalloc(DeviceType device, int deviceId, size_t size);
__C __export infinirtStatus_t *deviceMallocAsync(DeviceType device, int deviceId, size_t size, stream_t stream);
__C __export infinirtStatus_t deviceFree(void *ptr, DeviceType device, int deviceId);
__C __export infinirtStatus_t deviceFreeAsync(void *ptr, DeviceType device, int deviceId, stream_t stream);
__C __export infinirtStatus_t memcpyH2DAsync(void *dst, const void *src, size_t size, DeviceType device, int deviceId, stream_t stream);
__C __export infinirtStatus_t memcpyD2H(void *dst, const void *src, size_t size, DeviceType device, int deviceId);

#endif
