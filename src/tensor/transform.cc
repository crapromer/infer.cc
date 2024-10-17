#include "../tensor.h"
#include "../utils.h"
#include <algorithm>
#include <numeric>
#include <vector>


std::shared_ptr<Tensor> Tensor::slice_impl(const std::vector<SliceParams>& slices) const {
    std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>();
    
    auto new_shape = std::vector<index_t>(this->_shape);
    size_t offset = 0;

    for (const auto& slice : slices) {
        ASSERT(this->_shape[slice.dim] >= slice.start + slice.len);
        new_shape[slice.dim] = slice.len;
        offset += slice.start * this->_strides[slice.dim];
    }

    tensor->_dtype = this->_dtype;
    tensor->_shape = new_shape;
    tensor->_strides = std::vector<stride_t>(this->_strides);
    
    tensor->_data = static_cast<char *>(this->_data) + offset * dt_size(this->_dtype);
    
    tensor->_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                     dt_size(this->_dtype), std::multiplies<index_t>());
    tensor->storage = this->storage;
    infiniopCreateTensorDescriptor(&tensor->_desc, tensor->_shape.size(), tensor->_shape.data(),
                                   tensor->_strides.data(), dt_layout(tensor->_dtype));
    return tensor;
}


std::shared_ptr<Tensor> Tensor::slice(size_t dim, size_t start, size_t len) {
    return this->slice_impl({{dim, start, len}});
}

std::shared_ptr<Tensor const> Tensor::slice(size_t dim, size_t start, size_t len) const
{
    return this->slice_impl({{dim, start, len}});
}

std::shared_ptr<Tensor> Tensor::slice(const std::vector<SliceParams>& slices) {
    return this->slice_impl(slices);
}

std::shared_ptr<Tensor const> Tensor::slice(const std::vector<SliceParams>& slices) const
{
    return this->slice_impl(slices);
}

std::shared_ptr<Tensor> Tensor::dim_merge(size_t dim_start, size_t dim_end)
{
    ASSERT(dim_start <= dim_end && dim_end < this->_shape.size());
    if (dim_start == dim_end)
        return shared_from_this();

    auto new_shape = std::vector<index_t>();
    auto new_strides = std::vector<stride_t>();
    for (size_t i = 0; i < dim_start; i++)
    {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    for (size_t i = dim_start + 1; i <= dim_end; i++)
    {
        ASSERT_EQ(this->_strides[i - 1], this->_shape[i] * this->_strides[i]);
    }
    new_shape.push_back(std::accumulate(this->_shape.begin() + dim_start, this->_shape.begin() + dim_end + 1, 1, std::multiplies<index_t>()));
    new_strides.push_back(this->_strides[dim_end]);
    for (size_t i = dim_end + 1; i < this->_shape.size(); i++)
    {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    infiniopDestroyTensorDescriptor(this->_desc);
    infiniopCreateTensorDescriptor(&this->_desc, this->_shape.size(), this->_shape.data(),
                                   this->_strides.data(), dt_layout(this->_dtype));

    return shared_from_this();
}

std::shared_ptr<Tensor> Tensor::dim_split(size_t dim, const std::vector<size_t> &dims)
{
    ASSERT_EQ(this->_shape[dim], std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<index_t>()));
    auto new_shape = std::vector<index_t>();
    auto new_strides = std::vector<stride_t>();
    for (size_t i = 0; i < dim; i++)
    {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    for (size_t i = 0; i < dims.size(); i++)
    {
        new_shape.push_back(dims[i]);
        new_strides.push_back(this->_strides[dim] * this->_shape[dim] /
                              std::accumulate(dims.begin(),
                                              dims.begin() + i + 1, 1,
                                              std::multiplies<index_t>()));
    }
    for (size_t i = dim + 1; i < this->_shape.size(); i++)
    {
        new_shape.push_back(this->_shape[i]);
        new_strides.push_back(this->_strides[i]);
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    infiniopDestroyTensorDescriptor(this->_desc);
    infiniopCreateTensorDescriptor(&this->_desc, this->_shape.size(), this->_shape.data(),
                                   this->_strides.data(), dt_layout(this->_dtype));
    return shared_from_this();
}

std::shared_ptr<Tensor> Tensor::permute(const std::vector<size_t> &order) {
    ASSERT_EQ(this->_shape.size(), order.size());
    auto new_shape = std::vector<index_t>(order.size());
    auto new_strides = std::vector<stride_t>(order.size());
    for (size_t i = 0; i < order.size(); i++) {
        ASSERT(std::find(order.begin(), order.end(), i) != order.end());
        new_shape[i] = this->_shape[order[i]];
        new_strides[i] = this->_strides[order[i]];
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    infiniopDestroyTensorDescriptor(this->_desc);
    infiniopCreateTensorDescriptor(&this->_desc, this->_shape.size(), this->_shape.data(),
                                   this->_strides.data(), dt_layout(this->_dtype));
    return shared_from_this();
}
