#ifndef INFINIRT_ASCEND_H
#define INFINIRT_ASCEND_H
#include "../runtime.h"

#ifdef ENABLE_ASCEND_NPU
#define IMPL_WITH_ASCEND ;
#else
#define IMPL_WITH_ASCEND {  return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t initAscend() IMPL_WITH_ASCEND

infinirtStatus_t synchronizeAscendDevice(uint32_t deviceId) IMPL_WITH_ASCEND

infinirtStatus_t createAscendStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_ASCEND
infinirtStatus_t destoryAscendStream(infinirtStream_t stream) IMPL_WITH_ASCEND
infinirtStatus_t synchronizeAscendStream(infinirtStream_t stream) IMPL_WITH_ASCEND

infinirtStatus_t createAscendEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_ASCEND
infinirtStatus_t destoryAscendEvent(infinirtEvent_t event) IMPL_WITH_ASCEND
infinirtStatus_t waitAscendEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_ASCEND
infinirtStatus_t recordAscendEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_ASCEND
infinirtStatus_t queryAscendEvent(infinirtEvent_t event) IMPL_WITH_ASCEND
infinirtStatus_t synchronizeAscendEvent(infinirtEvent_t event) IMPL_WITH_ASCEND

infinirtStatus_t mallocAscend(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_ASCEND
infinirtStatus_t mallocAscendAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_ASCEND
infinirtStatus_t freeAscend(void *ptr, uint32_t deviceId) IMPL_WITH_ASCEND
infinirtStatus_t freeAscendAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream) IMPL_WITH_ASCEND
infinirtStatus_t memcpyHost2Ascend(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_ASCEND
infinirtStatus_t memcpyHost2AscendAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_ASCEND
infinirtStatus_t memcpyAscend2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_ASCEND
infinirtStatus_t memcpyAscendAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_ASCEND

#endif
