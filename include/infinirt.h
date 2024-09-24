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
    INFINIRT_STATUS_NOT_READY = 7,
} infinirtStatus_t;

struct InfinirtMemory
{
    void *ptr;
    size_t size;
    DeviceType device;
    uint32_t deviceId;
};
typedef struct InfinirtMemory *infinirtMemory_t;

// Stream
struct Stream;
typedef struct Stream *infinirtStream_t;
#define INFINIRT_NULL_STREAM nullptr
__C __export infinirtStatus_t infinirtStreamCreate(infinirtStream_t *pStream, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infinirtStreamDestroy(infinirtStream_t stream);
__C __export infinirtStatus_t infinirtGetRawStream(void** ptr, infinirtStream_t stream);

// Event
struct Event;
typedef struct Event *infinirtEvent_t;
__C __export infinirtStatus_t infinirtEventCreate(infinirtEvent_t *pEvent, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtEventQuery(infinirtEvent_t event);
__C __export infinirtStatus_t infinirtEventSynchronize(infinirtEvent_t event);
__C __export infinirtStatus_t infinirtEventDestroy(infinirtEvent_t event);
__C __export infinirtStatus_t infinirtStreamWaitEvent(infinirtEvent_t event, infinirtStream_t stream);

// Memory
__C __export infinirtStatus_t infinirtMalloc(infinirtMemory_t *pMemory, DeviceType device, uint32_t deviceId, size_t size);
__C __export infinirtStatus_t infinirtMallocAsync(infinirtMemory_t *pMemory, DeviceType device, uint32_t deviceId, size_t size, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtFree(infinirtMemory_t ptr);
__C __export infinirtStatus_t infinirtFreeAsync(infinirtMemory_t ptr, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtMemcpyH2DAsync(infinirtMemory_t dst, const void *src, size_t size, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtMemcpyD2H(void *dst, const infinirtMemory_t src, size_t size);
__C __export infinirtStatus_t infinirtMemcpyAsync(infinirtMemory_t dst, const infinirtMemory_t src, size_t size, infinirtStream_t stream);
#endif
