#pragma once
#include "wgpu_context.h"
#include "BindGroupEntry.hpp"
#include "BindGroupLayoutEntry.hpp"
class BindGroup {
public:
    BindGroup(WGPUDevice _device, const std::vector<BindGroupLayoutEntry>& _layoutEntries);

    void addEntry(BindGroupEntryBase* _entry);

    void finalize();

    WGPUBindGroupLayout getLayout() {
        return layout;
    }

    WGPUBindGroup getBindGroup() const {
        return bindGroup;
    }

private:
    WGPUDevice device;
    WGPUBindGroupLayout layout;
    WGPUBindGroup bindGroup;
    std::vector<WGPUBindGroupEntry> entries;
    std::vector<BindGroupEntryBase*> entryPtrs;
};

