#pragma once

#include "../include/infinirt.h"
#include <stdio.h>
#include <stdlib.h>

#define TEST_PASSED 0
#define TEST_FAILED 1

#define TEST_TRUE(expr)                                                        \
    do {                                                                       \
        if (!(expr)) {                                                         \
            printf("Test failed: %s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return TEST_FAILED;                                                \
        }                                                                      \
    } while (0)

#define TEST_EQUAL(a, b) TEST_TRUE((a) == (b))

#define RUN_TEST(test)                                                         \
    do {                                                                       \
        if (test == TEST_FAILED) {                                             \
            printf("\033[31m%s FAILED\033[0m\n", #test);                       \
        } else {                                                               \
            printf("%s PASSED\n", #test);                                      \
        }                                                                      \
    } while (0)

void test_tensor(DeviceType);
