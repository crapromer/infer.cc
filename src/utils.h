#ifndef INFINIRT_UTILS_H
#define INFINIRT_UTILS_H

#include <stdio.h>
#include <stdlib.h>

inline void assert_true(int expr, const char *msg, const char *file, int line)
{
    if (!expr)
    {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assert_true((expr), #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#endif
