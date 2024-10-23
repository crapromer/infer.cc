#include "../tensor.h"
#include "../utils.h"
#include <numeric>


std::shared_ptr<TensorDesc> TensorDesc::create(DataType dtype, const std::vector<index_t> &shape, const std::vector<stride_t> &strides) {
    std::shared_ptr<TensorDesc> desc = std::make_shared<TensorDesc>();
    infiniopCreateTensorDescriptor(&desc->_desc, shape.size(), shape.data(),
                                   strides.data(), dt_layout(dtype));
    return desc;
}

TensorDesc::~TensorDesc() {
    infiniopDestroyTensorDescriptor(this->_desc);
}


const std::vector<index_t> &Tensor::shape() const { return this->_shape; }
const std::vector<stride_t> &Tensor::strides() const { return this->_strides; }
size_t Tensor::ndim() const { return this->_shape.size(); }
DataType Tensor::dtype() const { return this->_dtype; }
size_t Tensor::byte_size() const { return this->_size; }
DeviceType Tensor::device_type() const { return this->storage->device; }
uint32_t Tensor::device_id() const { return this->storage->deviceId; }
Tensor::~Tensor() {}

std::shared_ptr<TensorDesc> Tensor::desc() const{ return TensorDesc::create(this->_dtype, this->_shape, this->_strides); }

std::shared_ptr<Tensor> Tensor::buffer(DataType dtype,
                                       const std::vector<index_t> &shape,
                                       DeviceType device, uint32_t device_id,
                                       infinirtStream_t stream) {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    tensor->_dtype = dtype;
    auto ndim = shape.size();
    if (shape.empty())
    {
        tensor->_shape = std::vector<index_t>{1};
        ndim = 1;
    }
    else
    {
        tensor->_shape = std::vector<index_t>(shape);
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), dt_size(dtype), std::multiplies<index_t>());
    auto strides = std::vector<stride_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor->_strides = strides;
    tensor->storage = Storage::createAsync(size, device, device_id, stream);
    tensor->_size = size;
    tensor->_data = tensor->storage->memory;
    infiniopCreateTensorDescriptor(&tensor->_desc, ndim, tensor->_shape.data(),
                                   strides.data(), dt_layout(dtype));
    return tensor;
}

std::shared_ptr<Tensor> Tensor::weight(void *data, DataType dtype,
                                       const std::vector<index_t> &shape,
                                       DeviceType device, uint32_t deviceId) {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    ;
    tensor->_dtype = dtype;
    auto ndim = shape.size();
    if (shape.empty())
    {
        tensor->_shape = std::vector<index_t>{1};
        ndim = 1;
    }
    else
    {
        tensor->_shape = std::vector<index_t>(shape);
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), dt_size(dtype), std::multiplies<index_t>());
    auto strides = std::vector<stride_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor->_strides = strides;
    if (device == DEVICE_CPU) {
        tensor->storage = Storage::create(size, device, deviceId);
        RUN_INFINI(infinirtMemcpyH2D(tensor->storage->memory, device, deviceId, data, size));
    } else {
        tensor->storage = Storage::create(size, device, deviceId);
        RUN_INFINI(infinirtMemcpyH2D(tensor->storage->memory, device, deviceId, data,
                               size));
    }
    tensor->_data = tensor->storage->memory;
    tensor->_size = size;
    infiniopCreateTensorDescriptor(&tensor->_desc, ndim, tensor->_shape.data(),
                                   strides.data(), dt_layout(dtype));
    return tensor;
}

void *Tensor::data_impl(index_t offset, infinirtStream_t stream) const {
    ASSERT(offset * dt_size(this->dtype()) < this->_size);

    if (this->storage->event != nullptr && infinirtEventQuery(this->storage->event) == INFINIRT_STATUS_NOT_READY) {
        if (stream == nullptr) {
            infinirtEventSynchronize(this->storage->event);
        } else {
            infinirtStreamWaitEvent(this->storage->event, stream);
        }
    }

    return (char *)(this->_data) + offset * dt_size(this->dtype());
}

void *Tensor::data(infinirtStream_t stream) {
    return this->data_impl(0, stream);
}

void const *Tensor::data(infinirtStream_t stream) const {
    return this->data_impl(0, stream);
}

void *Tensor::data(index_t offset, infinirtStream_t stream) {
    return this->data_impl(offset, stream);
}

void const *Tensor::data(index_t offset, infinirtStream_t stream) const {
    return this->data_impl(offset, stream);
}

void Tensor::copy_from(std::shared_ptr<Tensor const> src,
                       infiniopHandle_t handle, infinirtStream_t stream) {
    ASSERT_EQ(this->shape(), src->shape());
    ASSERT_EQ(this->dtype(), src->dtype());
    infiniopRearrangeDescriptor_t desc;
    void* raw_stream;
    if (stream != nullptr)
        infinirtGetRawStream(&raw_stream, stream);
    infiniopCreateRearrangeDescriptor(handle, &desc, src->desc()->get(),
                                      this->desc()->get());
    infiniopRearrange(desc, this->data(stream), src->data(stream), raw_stream);
    infiniopDestroyRearrangeDescriptor(desc);
    infinirtEventRecord(this->storage->event, stream);
}
