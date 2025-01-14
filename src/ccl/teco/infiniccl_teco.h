#ifndef INFINICCL_TECO_H_
#define INFINICCL_TECO_H_
#include "infiniccl.h"

#ifdef ENABLE_TECO_SDAA
#define IMPL_WITH_TECO ;
#else
#define IMPL_WITH_TECO { return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t infinicclSdaaCommInitAll(infinicclComm_t *comms, unsigned int numDevices, unsigned int const *deviceIDs) IMPL_WITH_TECO 
infinicclStatus_t infinicclSdaaCommDestroy(infinicclComm_t comm) IMPL_WITH_TECO 
infinicclStatus_t infinicclSdaaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype,
                              infinirtStream_t stream) IMPL_WITH_TECO

#endif /* INFINICCL_SDAA_H_ */