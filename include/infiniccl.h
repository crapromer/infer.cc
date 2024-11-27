#ifndef INFINI_CCL_H
#define INFINI_CCL_H
#include "infinirt.h"

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) &&                                                     \
    ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#else
#define __C
#endif

struct InfiniComm {
    DeviceType deviceType;
    unsigned int deviceID; // the actual device ID, not rank number
    void *comm;   // the actual communication object
};

typedef struct InfiniComm *infinicclComm_t;

typedef enum {
    INFINICCL_STATUS_SUCCESS = 0,
    INFINICCL_STATUS_EXECUTION_FAILED = 1,
    INFINICCL_STATUS_BAD_DEVICE = 2,
    INFINICCL_STATUS_DEVICE_NOT_SUPPORTED = 3,
    INFINICCL_STATUS_DEVICE_MISMATCH = 4,
    INFINICCL_STATUS_INVALID_ARGUMENT = 5,
    INFINICCL_STATUS_ILLEGAL_MEMORY_ACCESS = 6,
    INFINICCL_STATUS_BAD_DATATYPE = 7,
    INFINICCL_STATUS_COMMUNICATOR_UNINITIALIZED = 8,
} infinicclStatus_t;

#ifndef INFINI_DATATYPE
#define INFINI_DATATYPE
typedef enum {
    INFINI_BYTE = 0,
    INFINI_I8 = 1,
    INFINI_I16 = 2,
    INFINI_I32 = 3,
    INFINI_I64 = 4,
    INFINI_U8 = 5,
    INFINI_U16 = 6,
    INFINI_U32 = 7,
    INFINI_U64 = 8,
    INFINI_F8 = 9,
    INFINI_F16 = 10,
    INFINI_F32 = 11,
    INFINI_F64 = 12,
    INFINI_BF16 = 13,
    INFINI_BOOL = 14,
} InfiniDataType_t;
#endif

__C __export infinicclStatus_t infinicclCommInitAll(DeviceType deviceType,
                                                    infinicclComm_t *comms,
                                                    unsigned int numDevices,
                                                    unsigned int const *deviceIDs);
__C __export infinicclStatus_t infinicclCommDestroy(infinicclComm_t comm);
__C __export infinicclStatus_t infinicclAllReduceSum(
    infinicclComm_t comm, void *sendbuf, void *recvbuf, size_t count,
    InfiniDataType_t datatype, infinirtStream_t stream);

#endif
