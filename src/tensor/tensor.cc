#include "../tensor.h"
#include <numeric>
const std::vector<index_t> &Tensor::shape()
{
    return this->_shape;
}
const std::vector<stride_t> &Tensor::strides()
{
    return this->_strides;
}
size_t Tensor::ndim()
{
    return this->_shape.size();
}
DataType Tensor::dtype()
{
    return this->_dtype;
}

Tensor Tensor::buffer(DataType dtype, const std::vector<index_t> &shape, DeviceType device, uint32_t device_id, infinirtStream_t stream)
{
    Tensor tensor;
    tensor._dtype = dtype;
    auto ndim = shape.size();
    if (shape.empty())
    {
        tensor._shape = std::vector<index_t>{1};
        ndim = 1;
    }
    else
    {
        tensor._shape = std::vector<index_t>(shape);
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<index_t>());
    auto strides = std::vector<stride_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor._strides = strides;
    tensor.offset = 0;
    tensor.storage = Storage::createAsync(size, device, device_id, stream);
    return tensor;
}

Tensor Tensor::weight(void *data, DataType dtype, const std::vector<index_t> &shape, DeviceType device, uint32_t device_id)
{
    Tensor tensor;
    tensor._dtype = dtype;
    auto ndim = shape.size();
    if (shape.empty())
    {
        tensor._shape = std::vector<index_t>{1};
        ndim = 1;
    }
    else
    {
        tensor._shape = std::vector<index_t>(shape);
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<index_t>());
    auto strides = std::vector<stride_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor._strides = strides;
    tensor.offset = 0;
    tensor.storage = Storage::make(data, size);
    return tensor;
}

void *Tensor::data_ptr(infinirtStream_t stream)
{
    if (this->storage->event == nullptr)
        return static_cast<char*>(this->storage->memory->ptr) + this->offset;

    if (infinirtEventQuery(this->storage->event) == INFINIRT_STATUS_NOT_READY)
    {
        if (stream == nullptr)
        {
            infinirtEventSynchronize(this->storage->event);
            return static_cast<char*>(this->storage->memory->ptr) + this->offset;
        }
        else
        {
            infinirtStreamWaitEvent(this->storage->event, stream);
        }
    }

    return static_cast<char*>(this->storage->memory->ptr) + this->offset;
}
