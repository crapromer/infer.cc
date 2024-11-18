#include "../tensor.h"
#include "../utils.h"
#include <iostream>
#include <numeric>
#include <fstream>

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

size_t Tensor::data_offset() const {
    return (char *)(this->_data) - (char *)(this->storage->memory);
}

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
    tensor->storage = Storage::create(size, device, deviceId);
    RUN_INFINI(infinirtMemcpyH2D(tensor->storage->memory, device, deviceId,
                                 data, size));
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
    RUN_INFINI(infiniopCreateRearrangeDescriptor(
        handle, &desc, this->desc()->get(), src->desc()->get()));
    RUN_INFINI(infiniopRearrange(desc, this->data(stream), src->data(stream),
                                 raw_stream));
    RUN_INFINI(infiniopDestroyRearrangeDescriptor(desc));
    if (stream != nullptr) {
        if (this->storage->event == nullptr) {
            RUN_INFINI(infinirtEventCreate(&this->storage->event,
                                           this->storage->device,
                                           this->storage->deviceId));
        }
        RUN_INFINI(infinirtEventRecord(this->storage->event, stream));
    } else {
        RUN_INFINI(
            infinirtDeviceSynchronize(this->device_type(), this->device_id()));
    }
}

bool Tensor::is_contigous() const {
    auto ndim = this->ndim();
    auto shape = this->shape();
    auto strides = std::vector<stride_t>(ndim);
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    ASSERT_EQ(strides.size(), this->_strides.size());
    return std::equal(strides.begin(), strides.end(), this->_strides.begin());
}

template <typename T>
void print_data(T *data, const std::vector<index_t> &shape,
                const std::vector<stride_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (int i = 0; i < shape[dim]; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (int i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

template <>
void print_data(uint16_t const *data, const std::vector<index_t> &shape,
                const std::vector<stride_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (int i = 0; i < shape[dim]; i++) {
            std::cout << f16_to_f32(data[i * strides[dim]]) << " ";
        }
    } else if (dim < shape.size() - 1) {
        for (int i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
            std::cout << std::endl;
        }
    }
}

void Tensor::debug(const std::string &filename) const {
    RUN_INFINI(
        infinirtDeviceSynchronize(this->device_type(), this->device_id()));
    std::cout << "Tensor: "
              << "shape[ ";
    for (auto s : this->shape()) {
        std::cout << s << " ";
    }
    std::cout << "] strides[ ";
    for (auto s : this->strides()) {
        std::cout << s << " ";
    }
    std::cout << "] dtype=" << this->dtype()
              << " device=" << this->device_type()
              << " device_id=" << this->device_id() << std::endl;
    auto dtype = this->dtype();
    void const *cpu_data;
    if (this->device_type() != DEVICE_CPU) {
        void *cpu_memory = std::malloc(this->storage->size);
        RUN_INFINI(infinirtMemcpyD2H(cpu_memory, this->storage->memory,
                                     this->device_type(), this->device_id(),
                                     this->storage->size));
        cpu_data = cpu_memory;
    } else {
        cpu_data = this->data();
    }

    if (!filename.empty()){
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return;
        }
        outFile.write(reinterpret_cast<const char*>(cpu_data), this->storage->size);
        outFile.close();
        std::cout << "Data written to file: " << filename << "\n";
        return;
    }

    switch (dtype) {
    case DATA_TYPE_F16:
        print_data((uint16_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case DATA_TYPE_F32:
        print_data((float const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    case DATA_TYPE_U64:
        print_data((uint64_t const *)((char const *)cpu_data + data_offset()),
                   this->shape(), this->strides(), 0);
        break;
    default:
        PANIC("Unsupported data type");
    }
}

void Tensor::debug() const { this->debug(""); }
