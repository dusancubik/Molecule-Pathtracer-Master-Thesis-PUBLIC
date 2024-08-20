#include "computeRenderer.hpp"
#include "../KdTree/kdTreeRopes.hpp"

bool ComputeRenderer::initCamera() {
	camera = std::make_shared<Camera>(1280, 720, glm::vec3(0.f, 0.f, 3.0f));
	return true;
}



/*void ComputeRenderer::initBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {
	// Create a binding
	// 
	std::vector<WGPUBindGroupEntry> binding(1);

	createBindGroupEntry(binding[0],0, uniformBuffer,0, sizeof(CameraUBO) );

	
	// A bind group contains one or multiple bindings
	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;
	// There must be as many bindings as declared in the layout!
	bindGroupDesc.entryCount = 1;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);


}*/

void ComputeRenderer::initRaytracingBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {
	std::vector<WGPUBindGroupEntry> binding(4);

	//createBindGroupEntry(binding[0], 0, color_buffer_view, 0, ???);

	binding[0].nextInChain = nullptr;
	
	binding[0].binding = 0;
	
	binding[0].textureView = color_buffer_view;


	createBindGroupEntry(binding[1], 1, kdTreeStorageBuffer, 0, sizeof(KdTreeNodeSSBO) * kdTree->getTreeSSBOs().size());
	createBindGroupEntry(binding[2], 2, leavesStorageBuffer, 0, sizeof(LeafRopesUBO) * kdTree->getLeavesUBOs().size());
	createBindGroupEntry(binding[3], 3, spheresStorageBuffer, 0, sizeof(Sphere) * kdTree->getSphereSSBOs().size());
	// A bind group contains one or multiple bindings
	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;
	// There must be as many bindings as declared in the layout!
	bindGroupDesc.entryCount = 4;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}

void ComputeRenderer::initScreenBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {
	std::vector<WGPUBindGroupEntry> binding(2);

	//createBindGroupEntry(binding[0], 0, color_buffer_view, 0, ???);

	binding[0].nextInChain = nullptr;

	binding[0].binding = 0;

	binding[0].sampler = sampler;

	binding[1].nextInChain = nullptr;

	binding[1].binding = 1;
	
	binding[1].textureView = color_buffer_view;

	// A bind group contains one or multiple bindings
	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;
	// There must be as many bindings as declared in the layout!
	bindGroupDesc.entryCount = 2;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}

void ComputeRenderer::initUniforms() {
	//camera
	//createBuffer(uniformBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(CameraUBO), camera->getCameraUbo());
	//leaves
	createBuffer(leavesStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(LeafRopesUBO) * kdTree->getLeavesUBOs().size(), kdTree->getLeavesUBOs().data());
	//kd tree
	createBuffer(kdTreeStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(KdTreeNodeSSBO) * kdTree->getTreeSSBOs().size(), kdTree->getTreeSSBOs().data());
	//spheres
	createBuffer(spheresStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(Sphere) * kdTree->getSphereSSBOs().size(), kdTree->getSphereSSBOs().data());

}

void ComputeRenderer::render(WGPUTextureView &nextTexture) {
	//camera->updateCamera();
	//wgpuQueueWriteBuffer(queue, uniformBuffer, 0, camera->getCameraUbo(), sizeof(CameraUBO));
	//std::cout << "computer render()" << std::endl;
	WGPUCommandEncoderDescriptor commandEncoderDesc = {};
	commandEncoderDesc.nextInChain = nullptr;
	commandEncoderDesc.label = "Command Encoder";
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);

	WGPUComputePassDescriptor computePassDesc = {};
	computePassDesc.nextInChain = nullptr;
	
	WGPUComputePassEncoder raytracingPass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
	wgpuComputePassEncoderSetPipeline(raytracingPass, raytracingPipeline);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 0, raytracingBindGroup, 0, nullptr);

	wgpuComputePassEncoderDispatchWorkgroups(raytracingPass, 1024/8, 1024/8, 1);

	wgpuComputePassEncoderEnd(raytracingPass);

	WGPURenderPassDescriptor renderPassDesc = {};
	renderPassDesc.nextInChain = nullptr;

	WGPURenderPassColorAttachment renderPassColorAttachment = {};
	renderPassColorAttachment.view = nextTexture;
	renderPassColorAttachment.resolveTarget = nullptr;
	renderPassColorAttachment.loadOp = WGPULoadOp_Clear;
	renderPassColorAttachment.storeOp = WGPUStoreOp_Store;
	renderPassColorAttachment.clearValue = WGPUColor{ 0.05, 0.05, 0.75, 1.0 };
	renderPassDesc.colorAttachmentCount = 1;
	renderPassDesc.colorAttachments = &renderPassColorAttachment;

	WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);



	wgpuRenderPassEncoderSetPipeline(renderPass, screenPipeline);
	
	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, screenBindGroup, 0, nullptr);

	wgpuRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);

	wgpuRenderPassEncoderEnd(renderPass);

	WGPUCommandBufferDescriptor cmdBufferDesc = {};
	cmdBufferDesc.nextInChain = nullptr;
	cmdBufferDesc.label = "Command buffer";
	WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
	wgpuQueueSubmit(queue, 1, &commandBuffer);

	
	/*WGPURenderPassDescriptor renderPassDesc = {};
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

	
	wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
	// Draw 1 instance of a 3-vertices shape
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
	}*/
}

void ComputeRenderer::init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) {
	
	RendererBase::init(_spheres, _device, _queue, _swap_chain_default_format);
	//timestamp
	

	/*KdTree _kdTree;
	kdTree = &_kdTree;*/
	kdTree->construct(spheres);
	//inits
	initCamera();
	initDepthBuffer();
	initUniforms();
	//creating assets
	WGPUTextureDescriptor textureDesc{};
	textureDesc.nextInChain = nullptr;
	//textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.size = { 1280, 720, 1 };
	textureDesc.format = WGPUTextureFormat_RGBA8Unorm;
	textureDesc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
	textureDesc.sampleCount = 1;
	textureDesc.mipLevelCount = 1;
	color_buffer = wgpuDeviceCreateTexture(device,&textureDesc);

	WGPUTextureViewDescriptor textureViewDesc{};
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 1;
	color_buffer_view = wgpuTextureCreateView(color_buffer, &textureViewDesc);

	WGPUSamplerDescriptor samplerDescriptor{};
	samplerDescriptor.addressModeU = WGPUAddressMode_Repeat;
	samplerDescriptor.addressModeV = WGPUAddressMode_Repeat;
	samplerDescriptor.magFilter = WGPUFilterMode_Linear;
	samplerDescriptor.minFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.mipmapFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.maxAnisotropy = 1;

	sampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);

	screenShaderModule = ResourceManager::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\shaders\\compute\\screen_shader.wgsl", device);
	
	raytracingKernelModule = ResourceManager::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\shaders\\compute\\raytracing_kernel.wgsl", device);

	std::vector<WGPUBindGroupLayoutEntry> raytracingBindingLayout(4);


	createBindingLayout(0, 0, raytracingBindingLayout[0], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingBindingLayout[0].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
	raytracingBindingLayout[0].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
	raytracingBindingLayout[0].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
	raytracingBindingLayout[0].sampler = { 0 };

	createBindingLayout(1, sizeof(KdTreeNodeSSBO), raytracingBindingLayout[1], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Compute);
	createBindingLayout(2, sizeof(LeafRopesUBO), raytracingBindingLayout[2], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Compute);
	createBindingLayout(3, sizeof(Sphere), raytracingBindingLayout[3], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Compute);

	WGPUBindGroupLayoutDescriptor raytracingBindGroupLayoutDesc = {};
	raytracingBindGroupLayoutDesc.entryCount = 4;
	raytracingBindGroupLayoutDesc.entries = raytracingBindingLayout.data();//&bindingLayout;
	WGPUBindGroupLayout raytracingBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &raytracingBindGroupLayoutDesc);

	initRaytracingBindGroup(raytracingBindGroup, raytracingBindGroupLayout);

	WGPUComputePipelineDescriptor raytracingPipelineDesc = {};
	raytracingPipelineDesc.nextInChain = nullptr;

	raytracingPipelineDesc.compute.module = raytracingKernelModule;
	raytracingPipelineDesc.compute.entryPoint = "main";

	// Pipeline layout
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&raytracingBindGroupLayout;

	

	WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
	raytracingPipelineDesc.layout = layout;

	raytracingPipeline = wgpuDeviceCreateComputePipeline(device, &raytracingPipelineDesc);
	std::cout << "RaytracingPipeline compute pipeline: " << raytracingPipeline << std::endl;

	
	//screen shader

	std::vector<WGPUBindGroupLayoutEntry> screenBindingLayout(2);
	WGPUSamplerBindingLayout samplerBindingLayout{};
	samplerBindingLayout.type = WGPUSamplerBindingType_Filtering;
	createBindingLayout(0, 0, screenBindingLayout[0], WGPUBufferBindingType_Undefined, WGPUShaderStage_Fragment);
	screenBindingLayout[0].sampler = samplerBindingLayout;
	screenBindingLayout[0].sampler.nextInChain = nullptr;
	screenBindingLayout[0].sampler.type = WGPUSamplerBindingType_Filtering;
	WGPUTextureBindingLayout textureBindingLayout{};
	textureBindingLayout.sampleType = WGPUTextureSampleType_Float;
	createBindingLayout(1, 0, screenBindingLayout[1], WGPUBufferBindingType_Undefined, WGPUShaderStage_Fragment);
	screenBindingLayout[1].texture = textureBindingLayout;
	screenBindingLayout[1].texture.multisampled = false;
	screenBindingLayout[1].texture.sampleType = WGPUTextureSampleType_Float;
	screenBindingLayout[1].texture.viewDimension = WGPUTextureViewDimension_2D;

	WGPUBindGroupLayoutDescriptor screenBindGroupLayoutDesc = {};
	screenBindGroupLayoutDesc.entryCount = 2;
	screenBindGroupLayoutDesc.entries = screenBindingLayout.data();//&bindingLayout;
	WGPUBindGroupLayout screenBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &screenBindGroupLayoutDesc);

	initScreenBindGroup(screenBindGroup, screenBindGroupLayout);

	WGPURenderPipelineDescriptor screenPipelineDesc = {};
	screenPipelineDesc.nextInChain = nullptr;

	screenPipelineDesc.vertex.module = screenShaderModule;
	screenPipelineDesc.vertex.entryPoint = "vert_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	screenPipelineDesc.fragment = &fragmentState;
	fragmentState.module = screenShaderModule;
	fragmentState.entryPoint = "frag_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	screenPipelineDesc.multisample.mask = ~0u;
	screenPipelineDesc.multisample.alphaToCoverageEnabled = false;
	screenPipelineDesc.multisample.count = 1;

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
	//colorTarget.blend = &blendState;
	colorTarget.writeMask = WGPUColorWriteMask_All; 

	fragmentState.targetCount = 1;
	fragmentState.targets = &colorTarget;

	screenPipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	screenPipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
	screenPipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
	screenPipelineDesc.primitive.cullMode = WGPUCullMode_None;

	// Pipeline layout
	WGPUPipelineLayoutDescriptor screenLayoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	screenLayoutDesc.bindGroupLayoutCount = 1;
	screenLayoutDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&screenBindGroupLayout;

	WGPUPipelineLayout screenLayout = wgpuDeviceCreatePipelineLayout(device, &screenLayoutDesc);
	screenPipelineDesc.layout = screenLayout;

	screenPipeline = wgpuDeviceCreateRenderPipeline(device, &screenPipelineDesc);
	std::cout << "screenPipeline pipeline: " << screenPipeline << std::endl;
	std::cout << "KdTree Nodes size: " << kdTree->getTreeSSBOs().size() * sizeof(KdTreeNodeSSBO) << std::endl;
	std::cout << "Leaves size: " << kdTree->getLeavesUBOs().size() * sizeof(LeafRopesUBO) << std::endl;
	std::cout << "Spheres size: " << kdTree->getSphereSSBOs().size() * sizeof(Sphere) << std::endl;

}

