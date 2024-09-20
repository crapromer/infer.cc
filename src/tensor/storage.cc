#include "../tensor.h"

std::shared_ptr<Storage> Storage::make(void *data, size_t size){
    auto storage = std::make_shared<Storage>();
    infinirtMalloc(&storage->memory, DEVICE_CPU, 0, 0);
    infinirtMemcpyH2DAsync(storage->memory, data, size, INFINIRT_NULL_STREAM);
    storage->device = DEVICE_CPU;
    storage->deviceId = 0;
    storage->event = nullptr;
    return storage;
}

std::shared_ptr<Storage> Storage::create(size_t size, DeviceType device, uint32_t device_id)
{
    auto storage = std::make_shared<Storage>();
    infinirtMemory_t mem;
    infinirtMalloc(&mem, device, device_id, size);
    storage->memory = mem;
    storage->device = device;
    storage->deviceId = device_id;
    storage->event = nullptr;
    return storage;
}

std::shared_ptr<Storage> Storage::createAsync(size_t size, DeviceType device, uint32_t device_id, infinirtStream_t stream)
{
    auto storage = std::make_shared<Storage>();
    infinirtMemory_t mem;
    infinirtMallocAsync(&mem, device, device_id, size, stream);
    infinirtEvent_t event;
    infinirtEventCreate(&event, device, device_id);
    infinirtEventRecord(event, stream);
    storage->memory = mem;
    storage->device = device;
    storage->deviceId = device_id;
    storage->event = event;
    return storage;
}

Storage::~Storage()
{
    if (this->event)
    {
        if (infinirtEventQuery(this->event) == INFINIRT_STATUS_NOT_READY)
        {
            infinirtEventSynchronize(this->event);
        }
        infinirtEventDestroy(this->event);
    }
    if (this->memory)
        infinirtFree(this->memory);
}

