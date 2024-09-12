#ifndef INFER_TENSOR_H
#define INFER_TENSOR_H

#include "infini_infer.h"
#include <future>
#include <vector>

typedef uint64_t index_t;
typedef int64_t stride_t;

struct Storage
{
    std::future<void *> data_ptr;
    Device device;
    int device_id;
    void *event;
}

struct SliceArg
{
    index_t axis;
    index_t start;
    stride_t step;
    index_t len;
}

struct Tensor
{
    DataType dtype;
    index_t ndim;
    std::vector<index_t> shape;
    std::vector<stride_t> strides;
    index_t offset;
    std::shared_ptr<Storage> storage;

}

typedef struct Tensor *tensor_t;

tensor_t new_tensor(DataType dtype, const std::vector<index_t> &shape, Device device, int device_id);
tensor_t from_cpu(void *data, DataType dtype, const std::vector<index_t> &shape, Device device, int device_id);
void delete_tensor(tensor_t tensor);
tensor_t slice(tensor_t tensor, const SliceArg &arg);


#endif
