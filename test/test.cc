#include "test.h"

int main() {
#ifdef ENABLE_NV_GPU
    test_tensor(DEVICE_NVIDIA);
#endif
}
