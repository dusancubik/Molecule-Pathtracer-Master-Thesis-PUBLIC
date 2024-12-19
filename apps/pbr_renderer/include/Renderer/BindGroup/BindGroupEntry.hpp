/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: BindGroupEntry.hpp
 *
 *  Description:
 *  The BindGroupEntryBase class and its derived classes provide organized way to manage WGPUBindGroupEntry objects.
 *  Each derived class is specialized for a specific type of bind group entry: textures, buffers, or samplers.
 *  The BindGroupEntryBase class and its derived classes were generated by ChatGPT.
 * -----------------------------------------------------------------------------
 */
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