#include "../tensor.h"
#include "../utils.h"
#include <vector>
#include <numeric>

Tensor Tensor::slice_impl(size_t dim, size_t start, size_t len) const {
    Tensor tensor;
    ASSERT(this->_shape[dim] >= start + len);
    auto new_shape = std::vector<index_t>(this->_shape);
    new_shape[dim] = len;

    tensor._dtype = this->_dtype;
    tensor._shape = new_shape;
    tensor._strides = std::vector<stride_t>(this->_strides);
    tensor._data = (infinirtMemory_t)std::malloc(sizeof(InfinirtMemory));
    tensor._data->ptr = static_cast<char *>(this->_data->ptr) +
                        start * this->_strides[dim] * dt_size(this->_dtype);
    tensor._data->size =
        std::accumulate(new_shape.begin(), new_shape.end(),
                        dt_size(this->_dtype), std::multiplies<index_t>());
    tensor.storage = this->storage;
    return tensor;
}

Tensor Tensor::slice(size_t dim, size_t start, size_t len) {
    return this->slice_impl(dim, start, len);
}

const Tensor Tensor::slice(size_t dim, size_t start, size_t len) const
{
    return this->slice_impl(dim, start, len);
}

Tensor &Tensor::dim_merge(size_t dim_start, size_t dim_end)
{
    ASSERT(dim_start <= dim_end && dim_end < this->_shape.size());
    if (dim_start == dim_end)
        return *this;

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
    return *this;
}

Tensor &Tensor::dim_split(size_t dim, const std::vector<size_t> &dims)
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
        new_strides.push_back(this->_strides[dim] / std::accumulate(dims.begin(), dims.begin() + i, 1, std::multiplies<index_t>()));
    }
    for (size_t i = dim + 1; i < this->_shape.size(); i++)
    {
        new_shape.push_back(this->_shape[i]);
    }
    this->_shape = new_shape;
    this->_strides = new_strides;
    return *this;
}
