#include "infinirt_teco.h"
#include <sdaa_runtime.h>
#define SDAACHECK(error)                                                                           \
{                                                                                                  \
    sdaaError_t localError = error;                                                                \
    if ((localError != sdaaSuccess)) {                                                             \
      printf("error: '%s'(%d) from %s at %s:%d\n", sdaaGetErrorString(localError),               \
             localError, #error, __FUNCTION__, __LINE__);                                          \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
}

#define TECO_CALL(x)                                                           \
    do {                                                                       \
        sdaaError_t err = (x);                                                 \
        if (err != sdaaSuccess) {                                              \
            std::cerr << "Teco error: " << err << " in function " << __func__  \
                      << std::endl;                                            \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        sdaaError_t err = sdaaSetDevice(deviceId);                             \
        if (err != cudaSuccess) {                                              \
            std::cerr << "Teco set device " << deviceId << "error: " << err    \
                      << " in function " << __func__ << std::endl;             \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

inline sdaaStream_t getSdaaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<sdaaStream_t>(stream->stream);
}

infinirtStatus_t synchronizeTecoDevice(uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(sdaaDeviceSynchronize());
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createTecoStream(infinirtStream_t *pStream, uint32_t deviceId){
    SWITCH_DEVICE(deviceId);
    sdaaStream_t sdaa_stream;
    TECO_CALL(sdaaStreamCreate(&sdaa_stream));
    infinirtStream_t stream = new infinirtStream();
    stream->device = DEVICE_TECO;
    stream->device_id = deviceId;
    stream->stream = sdaa_stream;
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t destoryTecoStream(infinirtStream_t stream){
    SWITCH_DEVICE(stream->device_id);
    TECO_CALL(sdaaStreamDestroy(getSdaaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t synchronizeTecoStream(infinirtStream_t stream){
    SWITCH_DEVICE(stream->device_id);
    TECO_CALL(sdaaStreamSynchronize(getSdaaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createTecoEvent(infinirtEvent_t *pEvent, uint32_t deviceId){
    SWITCH_DEVICE(deviceId);
    sdaaEvent_t teco_event;
    TECO_CALL(sdaaEventCreate(&teco_event));
    infinirtEvent_t event = new infinirtEvent();
    event->device = DEVICE_TECO;
    event->device_id = deviceId;
    event->event = teco_event;
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t destoryTecoEvent(infinirtEvent_t event){
    SWITCH_DEVICE(event->device_id);
    TECO_CALL(sdaaEventDestroy(static_cast<sdaaEvent_t>(event->event)));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t waitTecoEvent(infinirtEvent_t event, infinirtStream_t stream){
    SWITCH_DEVICE(event->device_id);
    TECO_CALL(sdaaStreamWaitEvent(getSdaaStream(stream),
                                  static_cast<sdaaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t recordTecoEvent(infinirtEvent_t event, infinirtStream_t stream){
    SWITCH_DEVICE(event->device_id);
    TECO_CALL(sdaaEventRecord(static_cast<sdaaEvent_t>(event->event),
                              getSdaaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t queryTecoEvent(infinirtEvent_t event){
    SWITCH_DEVICE(event->device_id);
    sdaaError_t err = sdaaEventQuery(static_cast<sdaaEvent_t>(event->event));
    if (err != sdaaSuccess){
        return INFINIRT_STATUS_NOT_READY;
    }
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t synchronizeTecoEvent(infinirtEvent_t event){
    SWITCH_DEVICE(event->device_id);
    TECO_CALL(sdaaEventSynchronize(static_cast<sdaaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocTeco(void **pMemory, uint32_t deviceId, size_t size){
    SWITCH_DEVICE(deviceId);
    void *sdaa_ptr;
    TECO_CALL(sdaaMalloc(&sdaa_ptr, size));
    *pMemory = sdaa_ptr;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t mallocTecoAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream){
    SWITCH_DEVICE(deviceId);
    void *sdaa_ptr;
    TECO_CALL(sdaaMallocAsync(&sdaa_ptr, size, getSdaaStream(stream)));
    *pMemory = sdaa_ptr;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t mallocHostTeco(void **pMemory, uint32_t deviceId, size_t size){
    SWITCH_DEVICE(deviceId);
    void *host_ptr;
    TECO_CALL(sdaaMallocHost(&host_ptr, size));
    *pMemory = host_ptr;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t freeTeco(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaFree(ptr));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t freeTecoAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream){

}
infinirtStatus_t freeHostTeco(void *ptr, uint32_t deviceId){
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaFreeHost(ptr));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyHost2Teco(void *dst, uint32_t deviceId, const void *src, size_t size){
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaMemcpy(dst, src, size, sdaaMemcpyHostToDevice));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyHost2TecoAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream){
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaMemcpyAsync(dst, src, size, sdaaMemcpyHostToDevice,
                              getSdaaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyTeco2Host(void *dst, const void *src, uint32_t deviceId, size_t size){
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaMemcpy(dst, src, size, sdaaMemcpyDeviceToHost));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyTeco(void *dst, const void *src, uint32_t deviceId, size_t size){
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaMemcpy(dst, src, size, sdaaMemcpyDeviceToDevice));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyTecoAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream){
    SWITCH_DEVICE(deviceId);
    TECO_CALL(sdaaMemcpyAsync(dst, src, size, sdaaMemcpyDeviceToDevice,
                              getSdaaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}