#include "../tensor.h"

std::shared_ptr<Storage> Storage::host(void *data, size_t size){
    auto storage = std::make_shared<Storage>();
    storage->memory = data;
    storage->size = size;
    storage->device = DEVICE_CPU;
    storage->deviceId = 0;
    storage->event = nullptr;
    return storage;
}

std::shared_ptr<Storage> Storage::create(size_t size, DeviceType device, uint32_t device_id)
{
    auto storage = std::make_shared<Storage>();
    infinirtMalloc(&storage->memory, device, device_id, size);
    storage->size = size;
    storage->device = device;
    storage->deviceId = device_id;
    storage->event = nullptr;
    return storage;
}

std::shared_ptr<Storage> Storage::createAsync(size_t size, DeviceType device, uint32_t device_id, infinirtStream_t stream)
{
    auto storage = std::make_shared<Storage>();
    infinirtMallocAsync(&storage->memory, device, device_id, size, stream);
    infinirtEventCreate(&storage->event, device, device_id);
    infinirtEventRecord(storage->event, stream);
    storage->size = size;
    storage->device = device;
    storage->deviceId = device_id;
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
        infinirtFree(this->memory, this->device, this->deviceId);
}

