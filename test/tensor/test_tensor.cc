#include "../../include/infinirt.h"
#include "../../src/tensor.h"
#include "../test.h"
#include <vector>

int test_tensor_weight(DeviceType deviceType) {
    auto data =
        std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto result = std::vector<float>(10);
    auto tensor = Tensor::weight(data.data(), DATA_TYPE_F32,
                                 std::vector<index_t>({2, 5}), deviceType, 0);
    infinirtMemcpyD2H(result.data(), tensor.data(), tensor.byte_size());
    for (int i = 0; i < 10; i++) {
        TEST_EQUAL(result[i], data[i]);
    }

    return TEST_PASSED;
}

int test_tensor_buffer(DeviceType deviceType) {
    auto data =
        std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto result = std::vector<float>(10);
    auto tensor1 = Tensor::weight(data.data(), DATA_TYPE_F32,
                                  std::vector<index_t>({2, 5}), deviceType, 0);
    infinirtStream_t stream_data, stream_compute;
    infinirtStreamCreate(&stream_data, deviceType, 0);
    infinirtStreamCreate(&stream_compute, deviceType, 0);
    auto tensor2 = Tensor::buffer(DATA_TYPE_F32, std::vector<index_t>({2, 5}),
                                  deviceType, 0, stream_data);
    infinirtMemcpyAsync(tensor2.data(stream_compute), tensor1.data(stream_compute), tensor1.byte_size(),
                        stream_compute);
    infinirtMemcpyD2H(result.data(), tensor2.data(), tensor2.byte_size());

    for (int i = 0; i < 10; i++) {
        TEST_EQUAL(result[i], data[i]);
    }
    return TEST_PASSED;
}

int test_tensor_reshape(DeviceType deviceType) {
    auto data = std::vector<float>{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    auto result = std::vector<float>(12);
    auto tensor =
        Tensor::weight(data.data(), DATA_TYPE_F32,
                       std::vector<index_t>({2, 3, 2}), deviceType, 0);
    tensor.dim_merge(0, 1);
    TEST_EQUAL(tensor.shape(), std::vector<index_t>({6, 2}));
    TEST_EQUAL(tensor.strides(), std::vector<stride_t>({2, 1}));
    tensor.dim_split(0, {3, 2});
    TEST_EQUAL(tensor.shape(), std::vector<index_t>({3, 2, 2}));
    TEST_EQUAL(tensor.strides(), std::vector<stride_t>({4, 2, 1}));
    return TEST_PASSED;
}

int test_tensor_slice(DeviceType deviceType) {
    auto data = std::vector<float>{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    auto tensor =
        Tensor::weight(data.data(), DATA_TYPE_F32,
                       std::vector<index_t>({2, 3, 2}), deviceType, 0);
    auto tensor1 = tensor.slice(0, 1, 1);
    TEST_EQUAL(tensor1.shape(), std::vector<index_t>({1, 3, 2}));
    TEST_EQUAL(tensor1.strides(), std::vector<stride_t>({6, 2, 1}));
    TEST_EQUAL(tensor1.byte_size(), 6 * sizeof(float));
    TEST_EQUAL((char *)tensor1.data_ptr(),
               (char *)tensor.data_ptr() + 6 * sizeof(float));

    auto tensor2 = tensor.slice(1, 1, 1);
    TEST_EQUAL(tensor2.shape(), std::vector<index_t>({2, 1, 2}));
    TEST_EQUAL(tensor2.strides(), std::vector<stride_t>({6, 2, 1}));
    TEST_EQUAL(tensor2.byte_size(), 4 * sizeof(float));
    TEST_EQUAL((char *)tensor2.data_ptr(),
               (char *)tensor.data_ptr() + 2 * sizeof(float));

    auto tensor3 = tensor.slice(2, 0, 1);
    TEST_EQUAL(tensor3.shape(), std::vector<index_t>({2, 3, 1}));
    TEST_EQUAL(tensor3.strides(), std::vector<stride_t>({6, 2, 1}));
    TEST_EQUAL(tensor3.byte_size(), 6 * sizeof(float));
    TEST_EQUAL((char *)tensor3.data_ptr(),
               (char *)tensor.data_ptr());
    return TEST_PASSED;
}

void test_tensor(DeviceType deviceType) {
    RUN_TEST(test_tensor_weight(deviceType));
    RUN_TEST(test_tensor_buffer(deviceType));
    RUN_TEST(test_tensor_reshape(deviceType));
    RUN_TEST(test_tensor_slice(deviceType));
}
