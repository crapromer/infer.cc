#ifndef INFER_TENSOR_H
#define INFER_TENSOR_H

#include "infini_infer.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <string> 
typedef uint64_t index_t;
typedef int64_t stride_t;

struct Storage
{
    void *memory;
    size_t size;
    DeviceType device;
    uint32_t deviceId;
    infinirtEvent_t event;

    static std::shared_ptr<Storage> create(size_t size, DeviceType device, uint32_t device_id);
    static std::shared_ptr<Storage> createAsync(size_t size, DeviceType device, uint32_t device_id, infinirtStream_t stream = nullptr);
    ~Storage();
};

struct SliceParams {
    size_t dim;
    size_t start;
    size_t len;
};

class TensorDesc {
  private:
    infiniopTensorDescriptor_t _desc;

  public:
    static std::shared_ptr<TensorDesc>
    create(DataType dtype, const std::vector<index_t> &shape,
           const std::vector<stride_t> &strides);
    infiniopTensorDescriptor_t get() const { return _desc; };
    ~TensorDesc();
};

class Tensor: public std::enable_shared_from_this<Tensor>
{
private:
    DataType _dtype;
    std::vector<index_t> _shape;
    std::vector<stride_t> _strides;
    void *_data;
    index_t _size;
    std::shared_ptr<Storage> storage;
    infiniopTensorDescriptor_t _desc;

    void *data_impl(index_t offset, infinirtStream_t stream = nullptr) const;
    std::shared_ptr<Tensor>
    slice_impl(const std::vector<SliceParams> &slices) const;

  public:
    static std::shared_ptr<Tensor> buffer(DataType dtype,
                                          const std::vector<index_t> &shape,
                                          DeviceType device, uint32_t device_id,
                                          infinirtStream_t stream = nullptr);
    static std::shared_ptr<Tensor> weight(void *data, DataType dtype,
                                          const std::vector<index_t> &shape,
                                          DeviceType device,
                                          uint32_t device_id);
    std::shared_ptr<Tensor> slice(size_t dim, size_t start, size_t len);
    std::shared_ptr<Tensor const> slice(size_t dim, size_t start,
                                        size_t len) const;
    std::shared_ptr<Tensor> slice(const std::vector<SliceParams> &slices);
    std::shared_ptr<Tensor const>
    slice(const std::vector<SliceParams> &slices) const;
    std::shared_ptr<Tensor> dim_merge(size_t dim_start, size_t dim_end);
    std::shared_ptr<Tensor> dim_split(size_t dim, const std::vector<size_t> &dims);
    std::shared_ptr<Tensor> permute(const std::vector<size_t> &order);
    void *data(infinirtStream_t stream = nullptr);
    void const *data(infinirtStream_t stream = nullptr) const;
    void *data(index_t offset, infinirtStream_t stream = nullptr);
    void const *data(index_t offset, infinirtStream_t stream = nullptr) const;
    void copy_from(std::shared_ptr<Tensor const> src, infiniopHandle_t handle,
                   infinirtStream_t stream = nullptr);
    const std::vector<index_t> &shape() const;
    const std::vector<stride_t> &strides() const;
    size_t ndim() const;
    DataType dtype() const;
    std::shared_ptr<TensorDesc> desc() const;
    size_t byte_size() const;
    size_t data_offset() const;
    DeviceType device_type() const;
    uint32_t device_id() const;
    bool is_contigous() const;

    void debug(const std::string &filename="") const;

    ~Tensor();
};

inline size_t dt_size(DataType dtype) {
    switch (dtype) {
    case DATA_TYPE_F16:
        return 2;
    case DATA_TYPE_F32:
        return 4;
    case DATA_TYPE_U64:
        return 8;
    }
    PANIC("Unsupported data type");
    return 0;
}

inline DataLayout dt_layout(DataType dtype) {
    DataLayout layout;
    if (dtype == DATA_TYPE_F16) {
        layout = F16;
    } else if (dtype == DATA_TYPE_F32) {
        layout = F32;
    } else if (dtype == DATA_TYPE_U64) {
        layout = U64;
    }
    return layout;
}

#endif
