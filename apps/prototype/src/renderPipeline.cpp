#include "renderPipeline.hpp"
/*#include "ResourceManager.cpp"
#include "camera.cpp"
#include "timeStamp.cpp"
#include "kdTree.cpp"*/
bool RenderPipeline::initCamera() {
	camera = std::make_shared<Camera>(1280, 720, glm::vec3(0.f, 0.f, 3.0f));
	return true;
}

void RenderPipeline::initBindGroup() {
	// Create a binding
	// 
	std::vector<WGPUBindGroupEntry> binding(2);
	//WGPUBindGroupEntry binding = {};
	binding[0].nextInChain = nullptr;
	// The index of the binding (the entries in bindGroupDesc can be in any order)
	binding[0].binding = 0;
	// The buffer it is actually bound to
	binding[0].buffer = uniformBuffer;
	// We can specify an offset within the buffer, so that a single buffer can hold
	// multiple uniform blocks.
	binding[0].offset = 0;
	// And we specify again the size of the buffer.
	binding[0].size = sizeof(CameraUBO);

	std::cout << "bindgroup1\n";
	binding[1].nextInChain = nullptr;
	// The index of the binding (the entries in bindGroupDesc can be in any order)
	binding[1].binding = 1;
	// The buffer it is actually bound to
	binding[1].buffer = spheresStorageBuffer;
	// We can specify an offset within the buffer, so that a single buffer can hold
	// multiple uniform blocks.
	binding[1].offset = 0;
	// And we specify again the size of the buffer.
	binding[1].size = sizeof(Sphere) * spheres.size();
	std::cout << "bindgroup2\n";
	// A bind group contains one or multiple bindings
	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;
	// There must be as many bindings as declared in the layout!
	bindGroupDesc.entryCount = 2;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);


}

void RenderPipeline::initUniforms() {
	// Create uniform buffer
	WGPUBufferDescriptor bufferDesc{};
	bufferDesc.size = sizeof(CameraUBO);
	bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
	bufferDesc.mappedAtCreation = false;
	uniformBuffer = wgpuDeviceCreateBuffer(device, &bufferDesc);


	// Upload the initial value of the uniforms
	wgpuQueueWriteBuffer(queue, uniformBuffer, 0, camera->getCameraUbo(), sizeof(CameraUBO));

	// Create uniform buffer
	WGPUBufferDescriptor spheresBufferDesc{};
	spheresBufferDesc.nextInChain = nullptr;
	spheresBufferDesc.size = sizeof(Sphere) * spheres.size();
	spheresBufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
	spheresBufferDesc.mappedAtCreation = false;
	spheresStorageBuffer = wgpuDeviceCreateBuffer(device, &spheresBufferDesc);


	// Upload the initial value of the uniforms
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
	//front
	depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
	depthStencilState.stencilFront.failOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilFront.passOp = WGPUStencilOperation_Keep;

	//back
	depthStencilState.stencilBack.compare = WGPUCompareFunction_Always;
	depthStencilState.stencilBack.failOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilBack.depthFailOp = WGPUStencilOperation_Keep;
	depthStencilState.stencilBack.passOp = WGPUStencilOperation_Keep;


	//
	depthStencilState.depthCompare = WGPUCompareFunction_Less;
	depthStencilState.depthWriteEnabled = true;

	// Store the format in a variable as later parts of the code depend on it
	depthTextureFormat = WGPUTextureFormat_Depth24Plus;
	depthStencilState.format = depthTextureFormat;
	// Deactivate the stencil alltogether
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

	// The initial value of the depth buffer, meaning "far"
	depthStencilAttachment.depthClearValue = 1.0f;
	// Operation settings comparable to the color attachment
	depthStencilAttachment.depthLoadOp = WGPULoadOp_Clear;
	depthStencilAttachment.depthStoreOp = WGPUStoreOp_Store;
	// we could turn off writing to the depth buffer globally here
	depthStencilAttachment.depthReadOnly = false;

	// Stencil setup, mandatory but unused
	depthStencilAttachment.stencilClearValue = 0;
	depthStencilAttachment.stencilLoadOp = WGPULoadOp_Undefined;
	depthStencilAttachment.stencilStoreOp = WGPUStoreOp_Undefined;
	depthStencilAttachment.stencilReadOnly = true;

	//renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
	std::vector<WGPURenderPassTimestampWrite> timestampWritess = timestamp->getTimestamps();
	renderPassDesc.timestampWriteCount = 0;//2;
	renderPassDesc.timestampWrites = nullptr;//timestampWritess.data();
	wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 0);
	WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 0);

	//ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPass);
	// In its overall outline, drawing a triangle is as simple as this:
	// Select which render pipeline to use
	wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
	// Draw 1 instance of a 3-vertices shape
	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, bindGroup, 0, nullptr);

	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 0);

	//wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 0);
	wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 1);
	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 1);

	//wgpuCommandEncoderWriteTimestamp(encoder, timestamp->getQuerySet(), 1);
	/*wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, vertexBuffer, 0, vertexData.size() * sizeof(float));
	wgpuRenderPassEncoderSetIndexBuffer(renderPass, indexBuffer, WGPUIndexFormat_Uint16, 0, indexData.size() * sizeof(uint16_t));

	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, bindGroup, 0, nullptr);

	wgpuRenderPassEncoderDrawIndexed(renderPass, indexCount, 1, 0, 0, 0);*/
	//;
	//wgpuRenderPassEncoderWriteTimestamp(renderPass, timestamp->getQuerySet(), 1);
	//ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPass);
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
	timestamp = std::make_shared<Timestamp>(device);

	

	//inits
	initCamera();
	initDepthBuffer();
	


	shaderModule = ResourceManager::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\shaders\\raytracing.wgsl", device);
	std::cout << "Shader module: " << shaderModule << std::endl;

	std::cout << "Creating BASIC render pipeline..." << std::endl;


	//pipelineDesc.vertex.bufferCount = 1;
	//pipelineDesc.vertex.buffers = &vertexBufferLayout;
	//for raytracing -> no need buffer yet
	WGPURenderPipelineDescriptor pipelineDesc = {};
	pipelineDesc.nextInChain = nullptr;

	WGPUVertexAttribute vertexAttrib;
	std::vector<WGPUVertexAttribute> vertexAttribs(2);
	// [...]
	vertexAttribs[0].shaderLocation = 0;
	// Means vec2<f32> in the shader
	vertexAttribs[0].format = WGPUVertexFormat::WGPUVertexFormat_Float32x3;
	// Index of the first element
	vertexAttribs[0].offset = 0;

	vertexAttribs[1].shaderLocation = 1;
	// Means vec2<f32> in the shader
	vertexAttribs[1].format = WGPUVertexFormat::WGPUVertexFormat_Float32x3;
	// Index of the first element
	vertexAttribs[1].offset = 3 * sizeof(float);


	WGPUVertexBufferLayout vertexBufferLayout{};
	vertexBufferLayout.attributeCount = static_cast<uint32_t>(vertexAttribs.size());;
	vertexBufferLayout.attributes = vertexAttribs.data();
	vertexBufferLayout.arrayStride = 6 * sizeof(float);
	vertexBufferLayout.stepMode = WGPUVertexStepMode::WGPUVertexStepMode_Vertex;


	// Vertex shader
	pipelineDesc.vertex.module = shaderModule;
	pipelineDesc.vertex.entryPoint = "vs_main";
	pipelineDesc.vertex.constantCount = 0;
	pipelineDesc.vertex.constants = nullptr;

	pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
	pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
	pipelineDesc.primitive.cullMode = WGPUCullMode_None;

	// Fragment shader
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	pipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
	fragmentState.entryPoint = "fs_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	// Configure blend state
	WGPUBlendState blendState;
	// Usual alpha blending for the color:
	blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
	blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
	blendState.color.operation = WGPUBlendOperation_Add;
	// We leave the target alpha untouched:
	blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
	blendState.alpha.dstFactor = WGPUBlendFactor_One;
	blendState.alpha.operation = WGPUBlendOperation_Add;

	WGPUColorTargetState colorTarget = {};
	colorTarget.nextInChain = nullptr;
	colorTarget.format = swap_chain_default_format;
	colorTarget.blend = &blendState;
	colorTarget.writeMask = WGPUColorWriteMask_All; // We could write to only some of the color channels.

	// We have only one target because our render pass has only one output color
	// attachment.
	fragmentState.targetCount = 1;
	fragmentState.targets = &colorTarget;


	//pipelineDesc.depthStencil = &depthStencilState;

		// Multi-sampling
	// Samples per pixel
	pipelineDesc.multisample.count = 1;
	// Default value for the mask, meaning "all bits on"
	pipelineDesc.multisample.mask = ~0u;
	// Default value as well (irrelevant for count = 1 anyways)
	pipelineDesc.multisample.alphaToCoverageEnabled = false;

	// Create binding layout (don't forget to = Default)
	std::vector<WGPUBindGroupLayoutEntry> bindingLayout(2);

	//WGPUBindGroupLayoutEntry bindingLayout{};//Default();
	//Default
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
	// The binding index as used in the @binding attribute in the shader
	bindingLayout[0].binding = 0;
	// The stage that needs to access this resource
	bindingLayout[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;//ShaderStage::Vertex;
	bindingLayout[0].buffer.type = WGPUBufferBindingType_Uniform;//BufferBindingType::Uniform;
	bindingLayout[0].buffer.minBindingSize = sizeof(CameraUBO);

	//WGPUBindGroupLayoutEntry spheresBindingLayout{};//Default();
	//Default
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
	// The binding index as used in the @binding attribute in the shader
	bindingLayout[1].binding = 1;
	// The stage that needs to access this resource
	bindingLayout[1].visibility = WGPUShaderStage_Fragment;//ShaderStage::Vertex;
	bindingLayout[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage; //BufferBindingType::Uniform;
	bindingLayout[1].buffer.minBindingSize = sizeof(Sphere);
	//WGPUBufferBindingType_Storage

	// Create a bind group layout
	WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {};
	bindGroupLayoutDesc.entryCount = 2;
	bindGroupLayoutDesc.entries = bindingLayout.data();//&bindingLayout;
	bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bindGroupLayoutDesc);



	// Pipeline layout
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