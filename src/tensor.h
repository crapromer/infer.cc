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
};

struct SliceArg
{
    index_t axis;
    index_t start;
    stride_t step;
    index_t len;
};

class Tensor
{
private:
    DataType dtype;
    std::vector<index_t> shape;
    std::vector<stride_t> strides;
    index_t offset;
    std::shared_ptr<Storage> storage;

public:
    Tensor new_tensor(DataType dtype, const std::vector<index_t> &shape, Device device, int device_id);
    Tensor from_cpu(void *data, DataType dtype, const std::vector<index_t> &shape, Device device, int device_id);
    Tensor slice(const SliceArg &arg);
    Tensor &reshape(Tensor &tensor, const std::vector<index_t> &shape);
    void *data_ptr();
    const std::vector<index_t> shape();
    const std::vector<stride_t> strides();
    size_t ndim();
    DataType dtype();
    ~Tensor();
};

#endif
