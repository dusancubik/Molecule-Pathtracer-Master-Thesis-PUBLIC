#include "../../../include/Renderer/BindGroup/BindGroup.hpp"


BindGroup::BindGroup(WGPUDevice _device, const std::vector<BindGroupLayoutEntry>& _layoutEntries) : device(_device) {
    std::vector<WGPUBindGroupLayoutEntry> entries;
    for (const auto& le : _layoutEntries) {
        entries.push_back(le.getLayoutEntry());
    }

    WGPUBindGroupLayoutDescriptor desc = {};
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    layout = wgpuDeviceCreateBindGroupLayout(_device, &desc);
}

void BindGroup::addEntry(BindGroupEntryBase* _entry) {
    entries.push_back(_entry->getEntry());
    entryPtrs.push_back(std::move(_entry));
}

void BindGroup::finalize() {
    WGPUBindGroupDescriptor desc = {};
    desc.layout = layout;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    bindGroup = wgpuDeviceCreateBindGroup(device, &desc);
}
