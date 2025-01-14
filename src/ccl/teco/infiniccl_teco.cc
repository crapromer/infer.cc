#include "infiniccl_teco.h"
#include "../../runtime/runtime.h"
#include <sdaa_runtime.h>
#include <iostream>
#include <tccl.h>
#include <vector>

#define TCCL_CALL(x)                                                           \
    do {                                                                       \
        tcclResult_t tcclErr = (x);                                            \
        if (tcclErr != tcclSuccess) {                                          \
            std::cerr << "TCCL error: " << tcclErr << " in function "          \
                      << __func__ << std::endl;                                \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        sdaaError_t err = sdaaSetDevice(deviceId);                             \
        if (err != sdaaSuccess) {                                              \
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


inline tcclDataType_t getSdaaDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return tcclFloat;
    case INFINI_F16:
        return tcclHalf;
    default:
        return tcclHalf;
    }
}

infinicclStatus_t infinicclSdaaCommInitAll(infinicclComm_t *comms, unsigned int numDevices, unsigned int const *deviceIDs){
    return INFINICCL_STATUS_SUCCESS;
} 
infinicclStatus_t infinicclSdaaCommDestroy(infinicclComm_t comm) {
    return INFINICCL_STATUS_SUCCESS;
}
infinicclStatus_t infinicclSdaaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype,
                              infinirtStream_t stream){
                                return INFINICCL_STATUS_SUCCESS;
                              }