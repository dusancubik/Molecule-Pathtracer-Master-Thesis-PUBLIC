#pragma once
#include "wgpu_context.h"

class BindGroupLayoutEntry {
public:
    BindGroupLayoutEntry(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUBufferBindingType bufferType, uint64_t minBindingSize) {
        entry.binding = binding;
        entry.visibility = stageFlags;
        entry.buffer.type = bufferType;
        entry.buffer.minBindingSize = minBindingSize;
    }

    BindGroupLayoutEntry(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUTextureFormat format, WGPUStorageTextureAccess access) {//storage texture
        entry.binding = binding;
        entry.visibility = stageFlags;
        entry.storageTexture.access = access;
        entry.storageTexture.format = format;
        entry.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
    }

    BindGroupLayoutEntry(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUTextureViewDimension viewDimension, WGPUTextureSampleType sampleType) {//sampled texture
        entry.binding = binding;
        entry.visibility = stageFlags;
        entry.texture.sampleType = sampleType;
        entry.texture.viewDimension = viewDimension;
        entry.texture.multisampled = false;
    }

    BindGroupLayoutEntry(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUSamplerBindingType samplerType) {
        entry.binding = binding;
        entry.visibility = stageFlags;
        entry.sampler.type = samplerType;
    }

    WGPUBindGroupLayoutEntry getLayoutEntry() const {
        return entry;
    }

private:
    WGPUBindGroupLayoutEntry entry = {};
};
