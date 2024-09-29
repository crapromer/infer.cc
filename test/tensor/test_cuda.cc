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
    TEST_EQUAL(tensor.dim_merge(1, 2).shape(), std::vector<index_t>({2, 6}));
    TEST_EQUAL(tensor.dim_split(1, {2, 3}).shape(),
               std::vector<index_t>({2, 2, 3}));
    return TEST_PASSED;
}

int test_tensor_slice(DeviceType deviceType) {
    auto data = std::vector<float>{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                   7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    auto result = std::vector<float>(12);
    auto tensor =
        Tensor::weight(data.data(), DATA_TYPE_F32,
                       std::vector<index_t>({2, 3, 2}), deviceType, 0);
    auto tensor1 = tensor.slice(0, 1, 2);
    auto tensor2 = tensor.slice(1, 1, 1);
    auto tensor3 = tensor.slice(2, 0, 1);

}

int main() {
    RUN_TEST(test_tensor_weight(DEVICE_NVIDIA));
    RUN_TEST(test_tensor_buffer(DEVICE_NVIDIA));
    RUN_TEST(test_tensor_reshape(DEVICE_NVIDIA));
}
