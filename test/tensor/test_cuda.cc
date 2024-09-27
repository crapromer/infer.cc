#include "../../include/infinirt.h"
#include "../../src/tensor.h"
#include "../test.h"
#include <vector>

int test_weight_tensor(DeviceType deviceType) {
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

int test_buffer_tensor(DeviceType deviceType) {
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

int main() {
    RUN_TEST(test_weight_tensor(DEVICE_NVIDIA));
    RUN_TEST(test_buffer_tensor(DEVICE_NVIDIA));
}