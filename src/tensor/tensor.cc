#include "../tensor.h"
#include "../utils.h"
#include <numeric>

const std::vector<index_t> &Tensor::shape() const { return this->_shape; }
const std::vector<stride_t> &Tensor::strides() const { return this->_strides; }
size_t Tensor::ndim() const { return this->_shape.size(); }
DataType Tensor::dtype() const { return this->_dtype; }

TensorDescriptorHolder::TensorDescriptorHolder(
    DataType dtype, const std::vector<index_t> &shape,
    const std::vector<stride_t> &strides) {
    DataLayout layout;
    if (dtype == DATA_TYPE_F16) {
        layout = F16;
    } else if (dtype == DATA_TYPE_F32) {
        layout = F32;
    }

    infiniopCreateTensorDescriptor(&_desc, shape.size(),
                                   std::vector(shape).data(),
                                   std::vector(strides).data(), layout);
};

TensorDescriptorHolder Tensor::desc() const {
    return TensorDescriptorHolder(this->_dtype, this->_shape, this->_strides);
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

void const *Tensor::data_ptr(infinirtStream_t stream) const {
    return static_cast<char const *>(this->data_ptr(stream));
}

void Tensor::copy_from(const Tensor &src, infiniopHandle_t handle, infinirtStream_t stream) {
    ASSERT_EQ(this->shape(), src.shape());
    ASSERT_EQ(this->dtype(), src.dtype());
    infiniopRearrangeDescriptor_t desc;
    void* raw_stream;
    if (stream != nullptr)
        infinirtGetRawStream(&raw_stream, stream);
    infiniopCreateRearrangeDescriptor(handle, &desc, src.desc().get(), this->desc().get());
    infiniopRearrange(desc, this->data_ptr(stream), src.data_ptr(stream), raw_stream);
    infiniopDestroyRearrangeDescriptor(desc);
}
