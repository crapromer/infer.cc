#ifndef INFER_TENSOR_H
#define INFER_TENSOR_H

#include "infini_infer.h"
#include <future>
#include <vector>

typedef uint64_t index_t;
typedef int64_t stride_t;

struct Storage
{
    infinirtMemory_t memory;
    DeviceType device;
    uint32_t deviceId;
    infinirtEvent_t event;

    static std::shared_ptr<Storage> make(void *data, size_t size);
    static std::shared_ptr<Storage> create(size_t size, DeviceType device, uint32_t device_id);
    static std::shared_ptr<Storage> createAsync(size_t size, DeviceType device, uint32_t device_id, infinirtStream_t stream = nullptr);
    ~Storage();
};

class TensorDescriptorHolder {
  private:
    infiniopTensorDescriptor_t _desc;

  public:
    TensorDescriptorHolder(DataType dtype, const std::vector<index_t> &shape,
                           const std::vector<stride_t> &strides);
    infiniopTensorDescriptor_t get() const { return _desc; }
    ~TensorDescriptorHolder() { infiniopDestroyTensorDescriptor(_desc); }
};

class Tensor
{
private:
    DataType _dtype;
    std::vector<index_t> _shape;
    std::vector<stride_t> _strides;
    size_t offset;
    std::shared_ptr<Storage> storage;
    infiniopTensorDescriptor_t _desc;

  public:
    static Tensor buffer(DataType dtype, const std::vector<index_t> &shape, DeviceType device, uint32_t device_id, infinirtStream_t stream = nullptr);
    static Tensor weight(void *data, DataType dtype, const std::vector<index_t> &shape, DeviceType device, uint32_t device_id);
    Tensor slice(size_t dim, size_t start, size_t len);
    const Tensor slice(size_t dim, size_t start, size_t len) const;
    Tensor &dim_merge(size_t dim_start, size_t dim_end);
    Tensor &dim_split(size_t dim, const std::vector<size_t> &dims);
    void *data_ptr(infinirtStream_t stream = nullptr);
    void const *data_ptr(infinirtStream_t stream = nullptr) const;
    void copy_from(const Tensor &src, infiniopHandle_t handle, infinirtStream_t stream = nullptr);
    const std::vector<index_t> &shape() const;
    const std::vector<stride_t> &strides() const;
    size_t ndim() const;
    DataType dtype() const;
    TensorDescriptorHolder desc() const;
    ~Tensor();
};

#endif
