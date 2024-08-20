#include "kdTreeRopesRenderPipeline.hpp"
#include "rendererBase.cpp"
#include "../KdTree/kdTreeRopes.cpp"
bool KdTreeRopesRenderPipeline::initCamera() {
	camera = std::make_shared<Camera>(1280, 720, glm::vec3(0.f, 0.f, 3.0f));
	return true;
}



void KdTreeRopesRenderPipeline::initBindGroup(WGPUBindGroup &bindGroup,WGPUBindGroupLayout bindGroupLayout) {
	// Create a binding
	// 
	std::vector<WGPUBindGroupEntry> binding(4);

	createBindGroupEntry(binding[0],0, uniformBuffer,0, sizeof(CameraUBO) );
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

void KdTreeRopesRenderPipeline::initUniforms() {
	//camera
	createBuffer(uniformBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(CameraUBO), camera->getCameraUbo());
	//leaves
	createBuffer(leavesStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(LeafRopesUBO) * kdTree->getLeavesUBOs().size(), kdTree->getLeavesUBOs().data());
	//kd tree
	createBuffer(kdTreeStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(KdTreeNodeSSBO) * kdTree->getTreeSSBOs().size(), kdTree->getTreeSSBOs().data());
	//spheres
	createBuffer(spheresStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(Sphere) * kdTree->getSphereSSBOs().size(), kdTree->getSphereSSBOs().data());

}

void KdTreeRopesRenderPipeline::render(WGPUTextureView &nextTexture) {
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
	}
}

void KdTreeRopesRenderPipeline::init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) {
	
	RendererBase::init(_spheres, _device, _queue, _swap_chain_default_format);
	//timestamp
	

	/*KdTree _kdTree;
	kdTree = &_kdTree;*/
	kdTree->construct(spheres);
	//inits
	initCamera();
	initDepthBuffer();
	


	shaderModule = ResourceManager::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\analyst\\shaders\\raytracing_kdtree_ropes.wgsl", device);
	std::cout << "Shader module: " << shaderModule << std::endl;

	std::cout << "Creating BASIC render pipeline..." << std::endl;


	//pipelineDesc.vertex.bufferCount = 1;
	//pipelineDesc.vertex.buffers = &vertexBufferLayout;
	//for raytracing -> no need buffer yet
	WGPURenderPipelineDescriptor pipelineDesc = {};
	pipelineDesc.nextInChain = nullptr;

	

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
	std::vector<WGPUBindGroupLayoutEntry> bindingLayout(4);

	
	createBindingLayout(0, sizeof(CameraUBO), bindingLayout[0], WGPUBufferBindingType_Uniform, WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
	createBindingLayout(1, sizeof(KdTreeNodeSSBO), bindingLayout[1], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Fragment);
	createBindingLayout(2, sizeof(LeafRopesUBO), bindingLayout[2], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Fragment);
	createBindingLayout(3, sizeof(Sphere),bindingLayout[3], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Fragment);
	// Create a bind group layout
	WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {};
	bindGroupLayoutDesc.entryCount = 4;
	bindGroupLayoutDesc.entries = bindingLayout.data();//&bindingLayout;
	WGPUBindGroupLayout bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bindGroupLayoutDesc);



	// Pipeline layout
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&bindGroupLayout;

	WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
	pipelineDesc.layout = layout;

	pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
	std::cout << "Render pipeline: " << pipeline << std::endl;

	std::cout << "KdTree Nodes size: " << kdTree->getTreeSSBOs().size() * sizeof(KdTreeNodeSSBO) << std::endl;
	std::cout << "Leaves size: " << kdTree->getLeavesUBOs().size() * sizeof(LeafRopesUBO) << std::endl;
	std::cout << "Spheres size: " << kdTree->getSphereSSBOs().size() * sizeof(Sphere) << std::endl;
	initUniforms();
	initBindGroup(bindGroup,bindGroupLayout);
}

