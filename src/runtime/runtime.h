#ifndef __RUNTIME_H__
#define __RUNTIME_H__
#include "infinirt.h"
#include <stdint.h>
#include <stddef.h>

struct Stream{
    DeviceType device;
    uint32_t device_id;
    void* stream;
};

struct Event{
    DeviceType device;
    uint32_t device_id;
    void* event;
};
#endif
