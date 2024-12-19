#include "rendererBase.hpp"

void RendererBase::init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) {
	spheres = _spheres;
	device = _device;
	queue = _queue;
	swap_chain_default_format = _swap_chain_default_format;
	timestamp = std::make_shared<Timestamp<WGPURenderPassTimestampWrite>>(device);
}

void RendererBase::createBuffer(WGPUBuffer &buffer,WGPUBufferUsageFlags flags,uint64_t size,const void *data) {
	WGPUBufferDescriptor bufferDesc{};
	bufferDesc.size = size;
	bufferDesc.usage = flags;
	bufferDesc.mappedAtCreation = false;
	buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);


	
	wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
}



void RendererBase::createBindGroupEntry(WGPUBindGroupEntry& bindGroupEntry, uint32_t binding, WGPUBuffer buffer, uint64_t offset, uint64_t size) {
	bindGroupEntry.nextInChain = nullptr;
	
	bindGroupEntry.binding = binding;
	
	bindGroupEntry.buffer = buffer;
	
	bindGroupEntry.offset = offset;
	
	bindGroupEntry.size = size;
}


void RendererBase::createBindingLayout(uint32_t binding, uint64_t minBindingSize, WGPUBindGroupLayoutEntry& bindingLayout, WGPUBufferBindingType bufferType, WGPUShaderStageFlags shaderFlags) {

	

	bindingLayout.buffer.nextInChain = nullptr;
	bindingLayout.buffer.type = WGPUBufferBindingType_Undefined;
	bindingLayout.buffer.hasDynamicOffset = false;

	bindingLayout.sampler.nextInChain = nullptr;
	bindingLayout.sampler.type = WGPUSamplerBindingType_Undefined;

	bindingLayout.storageTexture.nextInChain = nullptr;
	bindingLayout.storageTexture.access = WGPUStorageTextureAccess_Undefined;
	bindingLayout.storageTexture.format = WGPUTextureFormat_Undefined;
	bindingLayout.storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

	bindingLayout.texture.nextInChain = nullptr;
	bindingLayout.texture.multisampled = false;
	bindingLayout.texture.sampleType = WGPUTextureSampleType_Undefined;
	bindingLayout.texture.viewDimension = WGPUTextureViewDimension_Undefined;
	
	bindingLayout.binding = binding;
	
	bindingLayout.visibility = shaderFlags;
	bindingLayout.buffer.type = bufferType;
	bindingLayout.buffer.minBindingSize = minBindingSize;

	//return bindingLayout;
}

void RendererBase::readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata) {
	RendererBase* pThis = (RendererBase*)userdata;
	int64_t* times =
		(int64_t*)wgpuBufferGetConstMappedRange(pThis->timestamp->getStagingBuffer(), 0, sizeof(int64_t) * 2);
	//WGPUProcBufferGetMappedRange()
	//WGPUProcBufferGetConstMappedRange();
	//wgpuBufferGetCon

	if (times != nullptr) {
		std::cout << "Frametime: " << (times[1] - times[0]) << "\n";
		pThis->frameTimeNS = (times[1] - times[0]);
	}

	std::cout << "readBufferMap callback" << "\n";
	wgpuBufferUnmap(pThis->timestamp->getStagingBuffer());

}

void RendererBase::initDepthBuffer() {
	depthStencilState.format = WGPUTextureFormat_Undefined;
	depthStencilState.depthWriteEnabled = false;
	depthStencilState.depthCompare = WGPUCompareFunction_Always;
	depthStencilState.stencilReadMask = 0xFFFFFFFF;
	depthStencilState.stencilWriteMask = 0xFFFFFFFF;
	depthStencilState.depthBias = 0;
	depthStencilState.depthBiasSlopeScale = 0;
	depthStencilState.depthBiasClamp = 0;
	
	depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
	depthStencilState.stencilFront.failOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilFront.passOp = WGPUStencilOperation_Keep;

	
	depthStencilState.stencilBack.compare = WGPUCompareFunction_Always;
	depthStencilState.stencilBack.failOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilBack.passOp = WGPUStencilOperation_Keep;


	
	depthStencilState.depthCompare = WGPUCompareFunction_Less;
	depthStencilState.depthWriteEnabled = true;

	
	depthTextureFormat = WGPUTextureFormat_Depth24Plus;
	depthStencilState.format = depthTextureFormat;
	
	depthStencilState.stencilReadMask = 0;
	depthStencilState.stencilWriteMask = 0;

	WGPUTextureDescriptor depthTextureDesc{};
	depthTextureDesc.dimension = WGPUTextureDimension_2D;
	depthTextureDesc.format = depthTextureFormat;
	depthTextureDesc.mipLevelCount = 1;
	depthTextureDesc.sampleCount = 1;
	depthTextureDesc.size = { 1280, 720, 1 }; //TODO: fix
	depthTextureDesc.usage = WGPUTextureUsage_RenderAttachment;
	depthTextureDesc.viewFormatCount = 1;
	depthTextureDesc.viewFormats = &depthTextureFormat;
	depthTexture = wgpuDeviceCreateTexture(device, &depthTextureDesc);

	WGPUTextureViewDescriptor depthTextureViewDesc{};
	depthTextureViewDesc.aspect = WGPUTextureAspect_DepthOnly;
	depthTextureViewDesc.baseArrayLayer = 0;
	depthTextureViewDesc.arrayLayerCount = 1;
	depthTextureViewDesc.baseMipLevel = 0;
	depthTextureViewDesc.mipLevelCount = 1;
	depthTextureViewDesc.dimension = WGPUTextureViewDimension_2D;
	depthTextureViewDesc.format = depthTextureFormat;
	depthTextureView = wgpuTextureCreateView(depthTexture, &depthTextureViewDesc);
}