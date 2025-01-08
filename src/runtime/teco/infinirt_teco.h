#ifndef INFINIRT_TECO_H
#define INFINIRT_TECO_H
#include "../runtime.h"

#ifdef ENABLE_TECO_SDAA
#define IMPL_WITH_TECO ;
#else
#define IMPL_WITH_TECO {  return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t synchronizeTecoDevice(uint32_t deviceId) IMPL_WITH_TECO

infinirtStatus_t createTecoStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_TECO
infinirtStatus_t destoryTecoStream(infinirtStream_t stream) IMPL_WITH_TECO
infinirtStatus_t synchronizeTecoStream(infinirtStream_t stream) IMPL_WITH_TECO

infinirtStatus_t createTecoEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_TECO
infinirtStatus_t destoryTecoEvent(infinirtEvent_t event) IMPL_WITH_TECO
infinirtStatus_t waitTecoEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_TECO
infinirtStatus_t recordTecoEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_TECO
infinirtStatus_t queryTecoEvent(infinirtEvent_t event) IMPL_WITH_TECO
infinirtStatus_t synchronizeTecoEvent(infinirtEvent_t event) IMPL_WITH_TECO

infinirtStatus_t mallocTeco(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_TECO
infinirtStatus_t mallocTecoAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_TECO
infinirtStatus_t mallocHostTeco(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_TECO
infinirtStatus_t freeTeco(void *ptr, uint32_t deviceId) IMPL_WITH_TECO
infinirtStatus_t freeTecoAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream) IMPL_WITH_TECO
infinirtStatus_t freeHostTeco(void *ptr, uint32_t deviceId) IMPL_WITH_TECO
infinirtStatus_t memcpyHost2Teco(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_TECO
infinirtStatus_t memcpyHost2TecoAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_TECO
infinirtStatus_t memcpyTeco2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_TECO
infinirtStatus_t memcpyTeco(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_TECO
infinirtStatus_t memcpyTecoAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_TECO
#endif
