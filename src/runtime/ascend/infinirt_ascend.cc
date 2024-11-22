#include "infinirt_ascend.h"
#include <acl/acl.h>
#include <mutex>

#define ACL_CALL(x)                                                            \
    do {                                                                       \
        aclError err = (x);                                                    \
        if (err != ACL_SUCCESS) {                                              \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        aclError err = aclrtSetDevice(deviceId);                               \
        if (err != ACL_SUCCESS) {                                              \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

std::once_flag acl_init_flag;

infinirtStatus_t initAscend(){
    aclError _err = ACL_SUCCESS;
    std::call_once(acl_init_flag, [&_err]() {
        _err = aclInit(nullptr);
    });
    ACL_CALL(_err);
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeAscendDevice(uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtSynchronizeDevice());
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createAscendStream(infinirtStream_t *pStream,
                                    uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    aclrtStream acl_stream;
    ACL_CALL(aclrtCreateStreamWithConfig(&acl_stream, 0, ACL_STREAM_FAST_LAUNCH));
    infinirtStream_t stream = new infinirtStream();
    stream->device = DEVICE_ASCEND;
    stream->device_id = deviceId;
    stream->stream = acl_stream;
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryAscendStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    ACL_CALL(aclrtDestroyStream(stream->stream));
    delete stream;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeAscendStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    ACL_CALL(aclrtSynchronizeStream(stream->stream));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createAscendEvent(infinirtEvent_t *pEvent, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    aclrtEvent acl_event;
    ACL_CALL(aclrtCreateEvent(&acl_event));
    infinirtEvent_t event = new infinirtEvent();
    event->device = DEVICE_ASCEND;
    event->device_id = deviceId;
    event->event = acl_event;
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryAscendEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    ACL_CALL(aclrtDestroyEvent(event->event));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t waitAscendEvent(infinirtEvent_t event,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    ACL_CALL(aclrtStreamWaitEvent(stream->stream, event->event));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t recordAscendEvent(infinirtEvent_t event,
                                   infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    ACL_CALL(aclrtRecordEvent(event->event, stream->stream));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t queryAscendEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    aclrtEventRecordedStatus status;
    aclError err = aclrtQueryEventStatus(event->event, &status);
    if (err == ACL_SUCCESS && ACL_EVENT_RECORDED_STATUS_COMPLETE == status) {
        return INFINIRT_STATUS_SUCCESS;
    }
    return INFINIRT_STATUS_NOT_READY;
}

infinirtStatus_t synchronizeAscendEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    ACL_CALL(aclrtSynchronizeEvent(event->event));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocAscend(void **pMemory, uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtMalloc(pMemory, size, ACL_MEM_MALLOC_HUGE_FIRST));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocAscendAsync(void **pMemory, uint32_t deviceId,
                                   size_t size, infinirtStream_t stream) {
    /// @todo Ascend does not support async malloc yet
    return mallocAscend(pMemory, deviceId, size);
}

infinirtStatus_t freeAscend(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtFree(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeAscendAsync(void *ptr, uint32_t deviceId,
                                 infinirtStream_t stream) {
    /// @todo Ascend does not support async free yet
    return freeAscend(ptr, deviceId);
}

infinirtStatus_t memcpyHost2Ascend(void *dst, uint32_t deviceId,
                                   const void *src, size_t size) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2AscendAsync(void *dst, uint32_t deviceId,
                                        const void *src, size_t size,
                                        infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE,
                              stream->stream));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyAscend2Host(void *dst, const void *src,
                                   uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyAscendAsync(void *dst, const void *src,
                                   uint32_t deviceId, size_t size,
                                   infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    ACL_CALL(aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                              stream->stream));
    return INFINIRT_STATUS_SUCCESS;
}
