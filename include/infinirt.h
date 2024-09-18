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
#include <stddef.h>
#include <stdint.h>

typedef enum
{
    DEVICE_CPU,
    DEVICE_NVIDIA,
    DEVICE_CAMBRICON,
} DeviceType;

typedef enum
{
    INFINIRT_STATUS_SUCCESS = 0,
    INFINIRT_STATUS_EXECUTION_FAILED = 1,
    INFINIRT_STATUS_BAD_DEVICE = 2,
    INFINIRT_STATUS_DEVICE_NOT_SUPPORTED = 3,
    INFINIRT_STATUS_DEVICE_MISMATCH = 4,
    INFINIRT_STATUS_INVALID_ARGUMENT = 5,
    INFINIRT_STATUS_ILLEGAL_MEMORY_ACCESS = 6,
} infinirtStatus_t;

struct Memory
{
    void *ptr;
    size_t size;
    DeviceType device;
    uint32_t deviceId;
};
typedef struct Memory *infiniMemory_t;

// Stream
struct Stream;
typedef struct Stream *infiniStream_t;
#define INFINIRT_NULL_STREAM nullptr
__C __export infinirtStatus_t infiniCreateStream(infiniStream_t *pStream, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infiniDestoryStream(infiniStream_t stream);

// Event
struct Event;
typedef struct Event *infiniEvent_t;
__C __export infinirtStatus_t infiniCreateEvent(infiniEvent_t *pEvent, infiniStream_t stream);
__C __export infinirtStatus_t infiniDestoryEvent(infiniEvent_t event);
__C __export infinirtStatus_t infiniWaitEvent(infiniEvent_t event, infiniStream_t stream);

// Memory
__C __export infinirtStatus_t infiniMalloc(infiniMemory_t *pMemory, DeviceType device, uint32_t deviceId, size_t size);
__C __export infinirtStatus_t infiniMallocAsync(infiniMemory_t *pMemory, DeviceType device, uint32_t deviceId, size_t size, infiniStream_t stream);
__C __export infinirtStatus_t infiniFree(infiniMemory_t ptr);
__C __export infinirtStatus_t infiniFreeAsync(infiniMemory_t ptr, infiniStream_t stream);
__C __export infinirtStatus_t infiniMemcpyH2DAsync(infiniMemory_t dst, const void* src, size_t size, infiniStream_t stream);
__C __export infinirtStatus_t infiniMemcpyD2H(void *dst, const infiniMemory_t src, size_t size);

#endif
