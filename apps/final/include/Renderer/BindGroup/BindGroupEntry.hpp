#pragma once
#include "wgpu_context.h"
class BindGroupEntryBase {
    public:

        virtual ~BindGroupEntryBase() = default;

        virtual WGPUBindGroupEntry getEntry()  {
            return entry;
        }

    protected:
        WGPUBindGroupEntry entry = {};
};

class TextureBindGroupEntry : public BindGroupEntryBase {
    public:
        TextureBindGroupEntry(uint32_t binding, WGPUTextureView view) {
            entry.binding = binding;
            entry.textureView = view;
        }
};

class BufferBindGroupEntry : public BindGroupEntryBase {
    public:
        BufferBindGroupEntry(uint32_t binding, WGPUBuffer buffer, uint64_t offset, uint64_t size) {
            entry.binding = binding;
            entry.buffer = buffer;
            entry.offset = offset;
            entry.size = size;
        }
};

class SamplerBindGroupEntry : public BindGroupEntryBase {
    public:
        SamplerBindGroupEntry(uint32_t binding, WGPUSampler sampler) {
            entry.binding = binding;
            entry.sampler = sampler;
        }
};
