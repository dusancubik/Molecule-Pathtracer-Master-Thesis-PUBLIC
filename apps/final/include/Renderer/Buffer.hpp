#pragma once
#include "wgpu_context.h"

class Buffer {
	public:
		Buffer(WGPUDevice _device, WGPUBufferUsageFlags _flags, uint64_t _size);
		void write(WGPUQueue _queue, const void* data, uint64_t offset);
		WGPUBuffer getBuffer() { return buffer; }
		uint64_t getSize() { return size; }
	private:
		WGPUBuffer buffer;
		uint64_t size;
};