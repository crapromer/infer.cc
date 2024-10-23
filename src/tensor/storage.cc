#include "../tensor.h"


std::shared_ptr<Storage> Storage::create(size_t size, DeviceType device, uint32_t device_id)
{
    auto storage = std::make_shared<Storage>();
    RUN_INFINI(infinirtMalloc(&storage->memory, device, device_id, size));
    storage->size = size;
    storage->device = device;
    storage->deviceId = device_id;
    storage->event = nullptr;
    return storage;
}

std::shared_ptr<Storage> Storage::createAsync(size_t size, DeviceType device, uint32_t device_id, infinirtStream_t stream)
{
    auto storage = std::make_shared<Storage>();
    RUN_INFINI(infinirtMallocAsync(&storage->memory, device, device_id, size, stream));
    RUN_INFINI(infinirtEventCreate(&storage->event, device, device_id));
    RUN_INFINI(infinirtEventRecord(storage->event, stream));
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
            RUN_INFINI(infinirtEventSynchronize(this->event));
        }
        RUN_INFINI(infinirtEventDestroy(this->event));
    }
    this->event = nullptr;
    if (this->memory)
        RUN_INFINI(infinirtFree(this->memory, this->device, this->deviceId));
}

