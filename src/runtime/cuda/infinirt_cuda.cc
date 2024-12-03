#include "infinirt_cuda.h"
#include "cuda_runtime.h"

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        cudaError_t err = cudaSetDevice(deviceId);                             \
        if (err != cudaSuccess) {                                              \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

inline cudaStream_t getCudaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<cudaStream_t>(stream->stream);
}

infinirtStatus_t synchronizeCudaDevice(uint32_t deviceId){
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaDeviceSynchronize());
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createCudaStream(infinirtStream_t *pStream,
                                  uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    cudaStream_t cuda_stream;
    CUDA_CALL(cudaStreamCreate(&cuda_stream));
    infinirtStream_t stream = new infinirtStream();
    stream->device = DEVICE_NVIDIA;
    stream->device_id = deviceId;
    stream->stream = cuda_stream;
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t destoryCudaStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    CUDA_CALL(cudaStreamDestroy(getCudaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeCudaStream(infinirtStream_t stream){
    SWITCH_DEVICE(stream->device_id);
    CUDA_CALL(cudaStreamSynchronize(getCudaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createCudaEvent(infinirtEvent_t *pEvent, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    cudaEvent_t cuda_event;
    CUDA_CALL(cudaEventCreate(&cuda_event));
    infinirtEvent_t event = new infinirtEvent();
    event->device = DEVICE_NVIDIA;
    event->device_id = deviceId;
    event->event = cuda_event;
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryCudaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    CUDA_CALL(cudaEventDestroy(static_cast<cudaEvent_t>(event->event)));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t waitCudaEvent(infinirtEvent_t event, infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    CUDA_CALL(cudaStreamWaitEvent(getCudaStream(stream),
                                  static_cast<cudaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t recordCudaEvent(infinirtEvent_t event,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    CUDA_CALL(cudaEventRecord(static_cast<cudaEvent_t>(event->event),
                              getCudaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t queryCudaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    CUDA_CALL(cudaEventQuery(static_cast<cudaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeCudaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    CUDA_CALL(cudaEventSynchronize(static_cast<cudaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocCuda(void **pMemory, uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    void *cuda_ptr;
    CUDA_CALL(cudaMalloc(&cuda_ptr, size));
    *pMemory = cuda_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocCudaAsync(void **pMemory, uint32_t deviceId, size_t size,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    void *cuda_ptr;
    CUDA_CALL(cudaMallocAsync(&cuda_ptr, size, getCudaStream(stream)));
    *pMemory = cuda_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocHostCuda(void **pMemory, uint32_t deviceId,
                                size_t size) {
    SWITCH_DEVICE(deviceId);
    void *host_ptr;
    CUDA_CALL(cudaMallocHost(&host_ptr, size));
    *pMemory = host_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeCuda(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaFree(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeCudaAsync(void *ptr, uint32_t deviceId,
                               infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaFreeAsync(ptr, getCudaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeHostCuda(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaFreeHost(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2Cuda(void *dst, uint32_t deviceId, const void *src,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2CudaAsync(void *dst, uint32_t deviceId,
                                      const void *src, size_t size,
                                      infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                              getCudaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyCuda2Host(void *dst, const void *src, uint32_t deviceId,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyCuda(void *dst, const void *src, uint32_t deviceId,
                            size_t size) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyCudaAsync(void *dst, const void *src, uint32_t deviceId,
                                 size_t size, infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                              getCudaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}
