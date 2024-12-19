#include "../../include/Renderer/Buffer.hpp"

Buffer::Buffer(WGPUDevice _device, WGPUBufferUsageFlags _flags, uint64_t _size) {
	WGPUBufferDescriptor bufferDesc{};
	bufferDesc.size = _size;
	bufferDesc.usage = _flags;
	bufferDesc.mappedAtCreation = false;

	size = _size;
	buffer = wgpuDeviceCreateBuffer(_device, &bufferDesc);

}

void Buffer::write(WGPUQueue _queue, const void* data, uint64_t offset) {
	wgpuQueueWriteBuffer(_queue, buffer, offset, data, size);
}

void Buffer::write(WGPUQueue _queue, const void* data, uint64_t offset, uint64_t _size) {
	wgpuQueueWriteBuffer(_queue, buffer, offset, data, _size);
}