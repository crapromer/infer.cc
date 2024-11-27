#include "infiniccl_cuda.h"
#include "../../runtime/runtime.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>

#define NCCL_CALL(x)                                                           \
    do {                                                                       \
        ncclResult_t ncclErr = (x);                                            \
        if (ncclErr != ncclSuccess) {                                          \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        cudaError_t err = cudaSetDevice(deviceId);                             \
        if (err != cudaSuccess) {                                              \
            return INFINICCL_STATUS_BAD_DEVICE;                                \
        }                                                                      \
    } while (0)

inline cudaStream_t getCudaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<cudaStream_t>(stream->stream);
}

inline ncclDataType_t getCudaDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return ncclFloat;
    case INFINI_F16:
        return ncclHalf;
    default:
        return ncclHalf;
    }
}

inline ncclComm_t getNcclComm(infinicclComm_t comm) {
    return static_cast<ncclComm_t>(comm->comm);
}

infinicclStatus_t infinicclCudaCommInitAll(infinicclComm_t *comms,
                                           unsigned int numDevices, unsigned int const *deviceIDs) {
    std::vector<ncclComm_t> ncclComms(numDevices);
    NCCL_CALL(ncclCommInitAll(ncclComms.data(), numDevices, (int const *)deviceIDs));

    for (int i = 0; i < numDevices; i++) {
        comms[i] = new InfiniComm{DEVICE_NVIDIA, deviceIDs[i], (void *)(ncclComms[i])};
    }
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclCudaCommDestroy(infinicclComm_t comm) {
    NCCL_CALL(ncclCommDestroy(getNcclComm(comm)));
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclCudaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                                            void *recvbuf, size_t count,
                                            InfiniDataType_t datatype,
                                            infinirtStream_t stream) {
    if (datatype != INFINI_F32 && datatype != INFINI_F16) {
        return INFINICCL_STATUS_BAD_DATATYPE;
    }
    SWITCH_DEVICE(comm->deviceID);
    NCCL_CALL(ncclAllReduce(sendbuf, recvbuf, count, getCudaDtype(datatype),
                            ncclSum, getNcclComm(comm), getCudaStream(stream)));
    return INFINICCL_STATUS_SUCCESS;
}
