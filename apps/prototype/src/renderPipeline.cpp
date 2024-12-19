#include "renderPipeline.hpp"
/*#include "ResourceManager.cpp"
#include "camera.cpp"
#include "timeStamp.cpp"
#include "kdTree.cpp"*/
bool RenderPipeline::initCamera() {
	camera = std::make_shared<PROTO_Camera>(1280, 720, glm::vec3(0.f, 0.f, 3.0f));
	return true;
}

void RenderPipeline::initBindGroup() {

	std::vector<WGPUBindGroupEntry> binding(2);

	binding[0].nextInChain = nullptr;
	
	binding[0].binding = 0;

	binding[0].buffer = uniformBuffer;

	binding[0].offset = 0;

	binding[0].size = sizeof(CameraUBO);

	std::cout << "bindgroup1\n";
	binding[1].nextInChain = nullptr;
	
	binding[1].binding = 1;
	
	binding[1].buffer = spheresStorageBuffer;

	binding[1].offset = 0;

	binding[1].size = sizeof(Sphere) * spheres.size();
	std::cout << "bindgroup2\n";

	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;

	bindGroupDesc.entryCount = 2;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);


}

void RenderPipeline::initUniforms() {

	WGPUBufferDescriptor bufferDesc{};
	bufferDesc.size = sizeof(CameraUBO);
	bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
	bufferDesc.mappedAtCreation = false;
	uniformBuffer = wgpuDeviceCreateBuffer(device, &bufferDesc);


	wgpuQueueWriteBuffer(queue, uniformBuffer, 0, camera->getCameraUbo(), sizeof(CameraUBO));


	WGPUBufferDescriptor spheresBufferDesc{};
	spheresBufferDesc.nextInChain = nullptr;
	spheresBufferDesc.size = sizeof(Sphere) * spheres.size();
	spheresBufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
	spheresBufferDesc.mappedAtCreation = false;
	spheresStorageBuffer = wgpuDeviceCreateBuffer(device, &spheresBufferDesc);


	wgpuQueueWriteBuffer(queue, spheresStorageBuffer, 0, spheres.data(), sizeof(Sphere) * spheres.size());
	
}

void RenderPipeline::initDepthBuffer() {
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

void RenderPipeline::render(WGPUTextureView &nextTexture) {
	camera->updateCamera();
	wgpuQueueWriteBuffer(queue, uniformBuffer, 0, camera->getCameraUbo(), sizeof(CameraUBO));

	WGPUCommandEncoderDescriptor commandEncoderDesc = {};
	commandEncoderDesc.nextInChain = nullptr;
	commandEncoderDesc.label = "Command Encoder";
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);

	WGPURenderPassDescriptor renderPassDesc = {};
	renderPassDesc.nextInChain = nullptr;

	WGPURenderPassColorAttachment renderPassColorAttachment = {};
	renderPassColorAttachment.view = nextTexture;
	renderPassColorAttachment.resolveTarget = nullptr;
	renderPassColorAttachment.loadOp = WGPULoadOp_Clear;
	renderPassColorAttachment.storeOp = WGPUStoreOp_Store;
	renderPassColorAttachment.clearValue = WGPUColor{ 0.05, 0.05, 0.05, 1.0 };
	renderPassDesc.colorAttachmentCount = 1;
	renderPassDesc.colorAttachments = &renderPassColorAttachment;

	WGPURenderPassDepthStencilAttachment depthStencilAttachment;
	depthStencilAttachment.view = depthTextureView;

	
	depthStencilAttachment.depthClearValue = 1.0f;
	
	depthStencilAttachment.depthLoadOp = WGPULoadOp_Clear;
	depthStencilAttachment.depthStoreOp = WGPUStoreOp_Store;
	
	depthStencilAttachment.depthReadOnly = false;

	
	depthStencilAttachment.stencilClearValue = 0;
	depthStencilAttachment.stencilLoadOp = WGPULoadOp_Undefined;
	depthStencilAttachment.stencilStoreOp = WGPUStoreOp_Undefined;
	depthStencilAttachment.stencilReadOnly = true;

	
	std::vector<WGPURenderPassTimestampWrite> timestampWritess = timestamp->getTimestamps();
	renderPassDesc.timestampWriteCount = 0;//2;
	renderPassDesc.timestampWrites = nullptr;//timestampWritess.data();
	wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 0);
	WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 0);


	wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);

	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, bindGroup, 0, nullptr);

	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 0);

	//wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 0);
	wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);

	wgpuRenderPassEncoderEnd(renderPass);
	//wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 1);

	wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 1);

	wgpuCommandEncoderResolveQuerySet(encoder, timestamp->getQuerySet(), 0, 2, timestamp->getQueryBuffer(), 0);




	//wgpuCommandEncoderResolveQuerySet(encoder, timestamp->getQuerySet(), 0, 2, timestamp->getQueryBuffer(), 0);
	WGPUCommandBufferDescriptor cmdBufferDesc = {};
	cmdBufferDesc.nextInChain = nullptr;
	cmdBufferDesc.label = "Command buffer";
	WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
	wgpuQueueSubmit(queue, 1, &commandBuffer);

	//wgpuCommandEncoderResolveQuerySet(encoder, timestamp->getQuerySet(), 0, 2, timestamp->getQueryBuffer(), 0);
	//copy timestamps
	commandEncoderDesc.nextInChain = nullptr;
	commandEncoderDesc.label = "Copy Command Encoder";
	WGPUCommandEncoder copyEncoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);
	if (wgpuBufferGetMapState(timestamp->getStagingBuffer()) == WGPUBufferMapState_Unmapped) {
		wgpuCommandEncoderCopyBufferToBuffer(copyEncoder, timestamp->getQueryBuffer(), 0, timestamp->getStagingBuffer(), 0, 2 * sizeof(int64_t));

		WGPUCommandBufferDescriptor copyCmdBufferDesc = {};
		copyCmdBufferDesc.nextInChain = nullptr;
		copyCmdBufferDesc.label = "Copy Command buffer";
		WGPUCommandBuffer copyCommandBuffer = wgpuCommandEncoderFinish(copyEncoder, &copyCmdBufferDesc);
		wgpuQueueSubmit(queue, 1, &copyCommandBuffer);

		//auto callback = std::bind(&MainApplication::readBufferMap,this);
		wgpuBufferMapAsync(timestamp->getStagingBuffer(), WGPUMapMode_Read, 0, sizeof(int64_t) * 2, &readBufferMap, this);
	}
}

void RenderPipeline::init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) {
	spheres = _spheres;

	device = _device;
	//context = _context;
	queue = _queue;
	swap_chain_default_format = _swap_chain_default_format;

	//timestamp
	timestamp = std::make_shared<Timestamp<WGPURenderPassTimestampWrite>>(device);

	

	//inits
	initCamera();
	initDepthBuffer();
	


	shaderModule = ResourceManager::loadShaderModule("shaders_prototype\\raytracing.wgsl", device);
	std::cout << "Shader module: " << shaderModule << std::endl;

	std::cout << "Creating BASIC render pipeline..." << std::endl;


	//pipelineDesc.vertex.bufferCount = 1;
	//pipelineDesc.vertex.buffers = &vertexBufferLayout;
	//for raytracing -> no need buffer yet
	WGPURenderPipelineDescriptor pipelineDesc = {};
	pipelineDesc.nextInChain = nullptr;

	WGPUVertexAttribute vertexAttrib;
	std::vector<WGPUVertexAttribute> vertexAttribs(2);
	
	vertexAttribs[0].shaderLocation = 0;
	
	vertexAttribs[0].format = WGPUVertexFormat::WGPUVertexFormat_Float32x3;
	
	vertexAttribs[0].offset = 0;

	vertexAttribs[1].shaderLocation = 1;
	
	vertexAttribs[1].format = WGPUVertexFormat::WGPUVertexFormat_Float32x3;
	
	vertexAttribs[1].offset = 3 * sizeof(float);


	WGPUVertexBufferLayout vertexBufferLayout{};
	vertexBufferLayout.attributeCount = static_cast<uint32_t>(vertexAttribs.size());;
	vertexBufferLayout.attributes = vertexAttribs.data();
	vertexBufferLayout.arrayStride = 6 * sizeof(float);
	vertexBufferLayout.stepMode = WGPUVertexStepMode::WGPUVertexStepMode_Vertex;


	
	pipelineDesc.vertex.module = shaderModule;
	pipelineDesc.vertex.entryPoint = "vs_main";
	pipelineDesc.vertex.constantCount = 0;
	pipelineDesc.vertex.constants = nullptr;

	pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
	pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
	pipelineDesc.primitive.cullMode = WGPUCullMode_None;

	
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	pipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
	fragmentState.entryPoint = "fs_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	
	WGPUBlendState blendState;
	
	blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
	blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
	blendState.color.operation = WGPUBlendOperation_Add;
	
	blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
	blendState.alpha.dstFactor = WGPUBlendFactor_One;
	blendState.alpha.operation = WGPUBlendOperation_Add;

	WGPUColorTargetState colorTarget = {};
	colorTarget.nextInChain = nullptr;
	colorTarget.format = swap_chain_default_format;
	colorTarget.blend = &blendState;
	colorTarget.writeMask = WGPUColorWriteMask_All; 
	fragmentState.targetCount = 1;
	fragmentState.targets = &colorTarget;



	pipelineDesc.multisample.count = 1;
	
	pipelineDesc.multisample.mask = ~0u;
	
	pipelineDesc.multisample.alphaToCoverageEnabled = false;

	
	std::vector<WGPUBindGroupLayoutEntry> bindingLayout(2);


	bindingLayout[0].buffer.nextInChain = nullptr;
	bindingLayout[0].buffer.type = WGPUBufferBindingType_Undefined;
	bindingLayout[0].buffer.hasDynamicOffset = false;

	bindingLayout[0].sampler.nextInChain = nullptr;
	bindingLayout[0].sampler.type = WGPUSamplerBindingType_Undefined;

	bindingLayout[0].storageTexture.nextInChain = nullptr;
	bindingLayout[0].storageTexture.access = WGPUStorageTextureAccess_Undefined;
	bindingLayout[0].storageTexture.format = WGPUTextureFormat_Undefined;
	bindingLayout[0].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

	bindingLayout[0].texture.nextInChain = nullptr;
	bindingLayout[0].texture.multisampled = false;
	bindingLayout[0].texture.sampleType = WGPUTextureSampleType_Undefined;
	bindingLayout[0].texture.viewDimension = WGPUTextureViewDimension_Undefined;
	
	bindingLayout[0].binding = 0;
	
	bindingLayout[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
	bindingLayout[0].buffer.type = WGPUBufferBindingType_Uniform;
	bindingLayout[0].buffer.minBindingSize = sizeof(CameraUBO);

	
	bindingLayout[1].buffer.nextInChain = nullptr;
	bindingLayout[1].buffer.type = WGPUBufferBindingType_Undefined;
	bindingLayout[1].buffer.hasDynamicOffset = false;

	bindingLayout[1].sampler.nextInChain = nullptr;
	bindingLayout[1].sampler.type = WGPUSamplerBindingType_Undefined;

	bindingLayout[1].storageTexture.nextInChain = nullptr;
	bindingLayout[1].storageTexture.access = WGPUStorageTextureAccess_Undefined;
	bindingLayout[1].storageTexture.format = WGPUTextureFormat_Undefined;
	bindingLayout[1].storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

	bindingLayout[1].texture.nextInChain = nullptr;
	bindingLayout[1].texture.multisampled = false;
	bindingLayout[1].texture.sampleType = WGPUTextureSampleType_Undefined;
	bindingLayout[1].texture.viewDimension = WGPUTextureViewDimension_Undefined;
	
	bindingLayout[1].binding = 1;
	
	bindingLayout[1].visibility = WGPUShaderStage_Fragment;
	bindingLayout[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage; 
	bindingLayout[1].buffer.minBindingSize = sizeof(Sphere);

	WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {};
	bindGroupLayoutDesc.entryCount = 2;
	bindGroupLayoutDesc.entries = bindingLayout.data();//&bindingLayout;
	bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bindGroupLayoutDesc);



	
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&bindGroupLayout;

	WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
	pipelineDesc.layout = layout;

	pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
	std::cout << "Render pipeline: " << pipeline << std::endl;

	initUniforms();
	initBindGroup();
}

void RenderPipeline::readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata) {
	RenderPipeline* pThis = (RenderPipeline*)userdata;
	int64_t* times =
		(int64_t*)wgpuBufferGetConstMappedRange(pThis->timestamp->getStagingBuffer(), 0, sizeof(int64_t) * 2);
	//WGPUProcBufferGetMappedRange()
	//WGPUProcBufferGetConstMappedRange();
	//wgpuBufferGetCon

	if (times != nullptr) {
		//std::cout << "Frametime: " << (times[1] - times[0]) << "\n";
		pThis->frameTimeNS = (times[1] - times[0]);
	}

	
	wgpuBufferUnmap(pThis->timestamp->getStagingBuffer());
	//mapped
}