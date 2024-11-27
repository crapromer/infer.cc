#include "test.h"
#include <iostream>

int main() {
#ifdef ENABLE_NV_GPU
    printf("Test tensor functions: Nvidia\n");
    test_tensor(DEVICE_NVIDIA);
    printf("Test CCL functions: Nvidia\n");
    test_ccl(DEVICE_NVIDIA);
#endif
#ifdef ENABLE_ASCEND_NPU
    printf("Test tensor functions: Ascend\n");
    test_tensor(DEVICE_ASCEND);
#endif
}
