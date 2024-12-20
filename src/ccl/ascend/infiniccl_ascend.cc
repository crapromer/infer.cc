#include "infiniccl_ascend.h"
#include "../../runtime/runtime.h"
#include <acl/acl.h>
#include <hccl.h>
#include <iostream>
#include <vector>

#define HCCL_CALL(x)                                                           \
    do {                                                                       \
        HcclResult err = (x);                                                  \
        if (err != HCCL_SUCCESS) {                                             \
            std::cerr << "HCCL error: " << err << " in function " << __func__  \
                      << std::endl;                                            \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        aclError err = aclrtSetDevice(deviceId);                               \
        if (err != ACL_SUCCESS) {                                              \
            std::cerr << "ACL set device " << deviceId << " Error: " << err    \
                      << " in function " << __func__ << std::endl;             \
            std::cerr << aclGetRecentErrMsg() << std::endl;                    \
            return INFINICCL_STATUS_BAD_DEVICE;                                \
        }                                                                      \
    } while (0)

inline aclrtStream getAscendStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<aclrtStream>(stream->stream);
}

inline HcclDataType getAscneDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return HCCL_DATA_TYPE_FP32;
    case INFINI_F16:
        return HCCL_DATA_TYPE_FP16;
    default:
        return HCCL_DATA_TYPE_FP16;
    }
}

inline HcclComm getHcclComm(infinicclComm_t comm) {
    return static_cast<HcclComm>(comm->comm);
}

infinicclStatus_t infinicclAscendCommInitAll(infinicclComm_t *comms,
                                             unsigned int numDevices,
                                             unsigned int const *deviceIDs) {
    std::vector<HcclComm> hcclComms(numDevices);
    // Ascend requires all devices to be initialized before calling HcclCommInitAll.
    for (unsigned int i = 0; i < numDevices; i++) {
        SWITCH_DEVICE(deviceIDs[i]);
    }
    HCCL_CALL(
        HcclCommInitAll(numDevices, (int32_t *)deviceIDs, hcclComms.data()));
    for (unsigned int i = 0; i < numDevices; i++) {
        comms[i] =
            new InfiniComm{DEVICE_ASCEND, deviceIDs[i], (void *)(hcclComms[i])};
    }
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclAscendCommDestroy(infinicclComm_t comm) {
    HCCL_CALL(HcclCommDestroy(getHcclComm(comm)));
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclAscendAllReduceSum(infinicclComm_t comm,
                                              void *sendbuf, void *recvbuf,
                                              size_t count,
                                              InfiniDataType_t datatype,
                                              infinirtStream_t stream) {
    HCCL_CALL(HcclAllReduce(sendbuf, recvbuf, (uint64_t)count,
                            getAscneDtype(datatype), HCCL_REDUCE_SUM,
                            getHcclComm(comm), getAscendStream(stream)));
    // Somehow program will hang here if stream is not synchronized.
    infinirtStreamSynchronize(stream);
    return INFINICCL_STATUS_SUCCESS;
}
