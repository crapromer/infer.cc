#include "infiniccl.h"
#include "../runtime/runtime.h"
#include "./ascend/infiniccl_ascend.h"
#include "./cuda/infiniccl_cuda.h"
#include "./teco/infiniccl_teco.h"

__C infinicclStatus_t infinicclCommInitAll(DeviceType deviceType,
                                           infinicclComm_t *comms,
                                           unsigned numDevices, unsigned const *deviceIDs) {
    switch (deviceType) {
    case DEVICE_NVIDIA:
        return infinicclCudaCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_ASCEND:
        return infinicclAscendCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_TECO:
        return infinicclSdaaCommInitAll(comms, numDevices, deviceIDs);
    default:
        return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinicclStatus_t infinicclCommDestroy(infinicclComm_t comm) {
    if (comm == nullptr) {
        return INFINICCL_STATUS_SUCCESS;
    }
    switch (comm->deviceType) {
    case DEVICE_NVIDIA:
        return infinicclCudaCommDestroy(comm);
    case DEVICE_ASCEND:
        return infinicclAscendCommDestroy(comm);
    case DEVICE_TECO:
        return infinicclSdaaCommDestroy(comm);
    default:
        return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinicclStatus_t infinicclAllReduceSum(infinicclComm_t comm, void *sendbuf,
                                            void *recvbuf, size_t count,
                                            InfiniDataType_t datatype,
                                            infinirtStream_t stream) {
    if (comm == nullptr) {
        return INFINICCL_STATUS_SUCCESS;
    }
    if (stream != nullptr && comm->deviceType != stream->device) {
        return INFINICCL_STATUS_DEVICE_MISMATCH;
    }
    switch (comm->deviceType) {
    case DEVICE_NVIDIA:
        return infinicclCudaAllReduceSum(comm, sendbuf, recvbuf, count,
                                         datatype, stream);
    case DEVICE_ASCEND:
        return infinicclAscendAllReduceSum(comm, sendbuf, recvbuf, count,
                                           datatype, stream);
    case DEVICE_TECO:
        return infinicclSdaaAllReduceSum(comm, sendbuf, recvbuf, count,
                                           datatype, stream);
    default:
        return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
