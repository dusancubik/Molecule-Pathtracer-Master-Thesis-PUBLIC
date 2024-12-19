#include "../../include/Renderer/Renderer.hpp"

void Renderer::init(WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format, Scene* _scene, Config* _config, BilateralFilterConfig* _bilateralFilterConfig, DebugConfig* _debugConfig) {

	device = _device;
	queue = _queue;
	swap_chain_default_format = _swap_chain_default_format;
    scene = _scene;

	bilateralFilterConfig = _bilateralFilterConfig;

	config = _config;
	config->currentIteration = 0;
	config->currentSample = 0;

	debugConfig = _debugConfig;
	debugConfig->pixelCoordinates.z = 0.;
	/*config.currentIteration = 0;
	config.currentSample = 0;

	config.maxIterations = 10;
	config.maxSamples = 128;*/

    cameraBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(CameraUBO));
	initMaterialSets();
	initPathtracingPipeline();
	initDebugPipeline();
	initAccumulationPipeline();
}

void Renderer::render(WGPUTextureView& nextTexture) {
	/*if (scene->isBVHChanged()) {
		if(bvhBuffer != NULL) bvhBuffer->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0);
		if(spheresBuffer != NULL) spheresBuffer->write(queue, scene->getBVH()->getSpheresGPU().data(), 0);
		scene->SetBVHChanged(false);
	}*/

	if (scene->getCamera()->didMove()) {
		sampleId = 0; iteration = 0; bilateralFilterConfig->accumulationFinished = 0;
	}
	scene->getCamera()->cameraUBO.position.w = iteration / 1.0f;
	//config->maxIterations = 2;
	config->currentIteration = iteration;
	config->currentSample = sampleId;
	config->time = glfwGetTime();
	float time = 0.f;
	config->uniformRandom = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (1.f)));
	configBuffer->write(queue, config, 0);
	materialsBuffer->write(queue, materialSets[materialSetIndex].getMaterials().data(), 0);
	debugConfigBuffer->write(queue, debugConfig, 0);
	
	//wgpuQueueWriteBuffer(queue, configBuffer, 0, &config, sizeof(Config));
	cameraBuffer->write(queue, scene->getCamera()->getCameraUbo(), 0);

	/*if (iteration < config->maxIterations - 1) {
		iteration++;
	}
	else if (iteration >= config->maxIterations - 1 && sampleId < config->maxSamples - 1) {
		iteration = 0;
		sampleId++;
	}
	else if (sampleId == config->maxSamples-1) {
		//accumulation finished
		bilateralFilterConfig->accumulationFinished = 1;
	}*/
	bilateralFilterConfigBuffer->write(queue, bilateralFilterConfig, 0);

	if (debugMode) {
		render_debug(nextTexture);
		return;
	}
	//if (iteration % 5 == 0) iteration = 0;
	
	WGPUCommandEncoderDescriptor commandEncoderDesc = {};
	commandEncoderDesc.nextInChain = nullptr;
	commandEncoderDesc.label = "Command Encoder";
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);

	WGPUComputePassDescriptor computePassDesc = {};
	computePassDesc.nextInChain = nullptr;

	WGPUComputePassEncoder raytracingPass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
	wgpuComputePassEncoderSetPipeline(raytracingPass, pathtracingPipeline);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 0, pathtracingDataBindGroup->getBindGroup(), 0, nullptr);
	//ping ponging texture
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 1, ((iteration % 2 == 0 || iteration == 0))  ? computeTexturesBindGroupAlpha->getBindGroup() : computeTexturesBindGroupBeta->getBindGroup(), 0, nullptr);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 2, scene->getSkyboxesContainer()->getCurrentSkyboxBindGroup()->getBindGroup(), 0, nullptr);
	//wgpuComputePassEncoderSetBindGroup(raytracingPass, 3, ((sampleId % 2 == 0)) ? debugCounterBindGroupAlpha->getBindGroup() : debugCounterBindGroupBeta->getBindGroup(), 0, nullptr);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 3, bvhDataBindGroup->getBindGroup(), 0, nullptr);
	wgpuComputePassEncoderDispatchWorkgroups(raytracingPass, 1280 / 8, 720 / 4, 1);

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



	wgpuRenderPassEncoderSetPipeline(renderPass, accumulationPipeline);

	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, configBindGroup->getBindGroup(), 0, nullptr);
	wgpuRenderPassEncoderSetBindGroup(renderPass, 1, ((iteration % 2 == 0) || iteration == 0)  ? sampleBindGroupAlpha->getBindGroup() : sampleBindGroupBeta->getBindGroup(), 0, nullptr);
	wgpuRenderPassEncoderSetBindGroup(renderPass, 2, ((config->currentSample % 2) == 0 || sampleId == 0) ? accumulationTexturesBindGroupAlpha->getBindGroup() : accumulationTexturesBindGroupBeta->getBindGroup(), 0, nullptr);
	//else wgpuRenderPassEncoderSetBindGroup(renderPass, 1, screenDataTextureBindGroup1, 0, nullptr);
	wgpuRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);

	wgpuRenderPassEncoderEnd(renderPass);

	if (iteration < config->maxIterations - 1) {
		iteration++;
	}
	else if (iteration >= config->maxIterations - 1 && sampleId < config->maxSamples - 1) {
		iteration = 0;
		sampleId++;
	}
	else if (sampleId == config->maxSamples - 1) {
		//accumulation finished
		bilateralFilterConfig->accumulationFinished = 1;
	}

	WGPUCommandBufferDescriptor cmdBufferDesc = {};
	cmdBufferDesc.nextInChain = nullptr;
	cmdBufferDesc.label = "Command buffer";
	WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
	wgpuQueueSubmit(queue, 1, &commandBuffer);
}

void Renderer::render_debug(WGPUTextureView& nextTexture) {
	WGPUCommandEncoderDescriptor commandEncoderDesc = {};
	commandEncoderDesc.nextInChain = nullptr;
	commandEncoderDesc.label = "Command Debug Encoder";
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);

	WGPUComputePassDescriptor computePassDesc = {};
	computePassDesc.nextInChain = nullptr;

	WGPUComputePassEncoder raytracingPass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
	wgpuComputePassEncoderSetPipeline(raytracingPass, debugComputePipeline);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 0, debugDataBindGroup->getBindGroup(), 0, nullptr);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 1, bvhDataBindGroup->getBindGroup(), 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(raytracingPass, 1280 / 8, 720 / 4, 1);

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



	wgpuRenderPassEncoderSetPipeline(renderPass, debugScreenPipeline);

	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, debugScreenBindGroup->getBindGroup(), 0, nullptr);
	//else wgpuRenderPassEncoderSetBindGroup(renderPass, 1, screenDataTextureBindGroup1, 0, nullptr);
	wgpuRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);

	wgpuRenderPassEncoderEnd(renderPass);


	WGPUCommandBufferDescriptor cmdBufferDesc = {};
	cmdBufferDesc.nextInChain = nullptr;
	cmdBufferDesc.label = "Command Debug buffer";
	WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
	wgpuQueueSubmit(queue, 1, &commandBuffer);

	
	
}

void Renderer::initPathtracingPipeline(){
	WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("resources\\Shaders\\path_bvh_accumulated.wgsl", device);

	

	//pathtracing data group
	std::vector<BindGroupLayoutEntry> layoutEntries = {
		//(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUTextureFormat format, WGPUStorageTextureAccess access)
		//BindGroupLayoutEntry(0,WGPUShaderStage_Compute,WGPUBufferBindingType_ReadOnlyStorage,sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size()),
		//BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUBufferBindingType_ReadOnlyStorage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size()),
		BindGroupLayoutEntry(0, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(CameraUBO)),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(Config)),
		BindGroupLayoutEntry(2, WGPUShaderStage_Compute, WGPUBufferBindingType_ReadOnlyStorage, sizeof(Material) /* materialSets[0].getMaterials().size()*/),
		BindGroupLayoutEntry(3,WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,128 * sizeof(DebugData)),
		BindGroupLayoutEntry(4, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(DebugConfig))
	};

	bvhBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size());
	bvhBuffer->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0);
	spheresBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(SphereGPU) *  scene->getBVH()->getSpheresGPU().size());
	spheresBuffer->write(queue, scene->getBVH()->getSpheresGPU().data(), 0);
	
	configBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(Config));
	configBuffer->write(queue,config,0);
	bilateralFilterConfigBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(BilateralFilterConfig));
	bilateralFilterConfigBuffer->write(queue, bilateralFilterConfig, 0);
	Texture* skyboxTexture = new Texture(device, 0, 0, WGPUTextureFormat_RGBA8Unorm, scene->getSkybox()->getCubemapData());//sampler inside
	materialsBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(Material) * materialSets[0].getMaterials().size());
	materialsBuffer->write(queue, materialSets[0].getMaterials().data(), 0);

	debugLineArrayBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 128 * sizeof(DebugData));
	debugConfigBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(DebugConfig));

	pathtracingDataBindGroup = new BindGroup(device,layoutEntries);

	//pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(0, bvhBuffer->getBuffer(), 0, bvhBuffer->getSize()));
	//pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(1, spheresBuffer->getBuffer(), 0, spheresBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(0, cameraBuffer->getBuffer(), 0, cameraBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(1, configBuffer->getBuffer(), 0, configBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(2, materialsBuffer->getBuffer(), 0, materialsBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(3, debugLineArrayBuffer->getBuffer(), 0, debugLineArrayBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(4, debugConfigBuffer->getBuffer(), 0, debugConfigBuffer->getSize()));
	//pathtracingDataBindGroup->addEntry(new TextureBindGroupEntry(4, skyboxTexture->getTextureView()));
	//pathtracingDataBindGroup->addEntry(new SamplerBindGroupEntry(5, skyboxTexture->getSampler()));

	pathtracingDataBindGroup->finalize();
	//bindGroup = pathtracingDataBindGroup->getBindGroup();

	colorTextureAlpha = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA8Unorm);
	colorTextureBeta = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA8Unorm);

	Texture* rayOriginTextureAlpha = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA32Float);
	Texture* rayOriginTextureBeta = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA32Float);

	Texture* rayDirectionTextureAlpha = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA32Float);
	Texture* rayDirectionTextureBeta = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA32Float);

	//pathtracing data group
	std::vector<BindGroupLayoutEntry> textureLayoutEntries = {
		//(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUTextureFormat format, WGPUStorageTextureAccess access)
		BindGroupLayoutEntry(0, WGPUShaderStage_Compute, WGPUTextureFormat_RGBA32Float, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUTextureFormat_RGBA32Float, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(2, WGPUShaderStage_Compute, WGPUTextureViewDimension_2D, WGPUTextureSampleType_UnfilterableFloat),	
		BindGroupLayoutEntry(3, WGPUShaderStage_Compute, WGPUTextureViewDimension_2D, WGPUTextureSampleType_UnfilterableFloat),
		//BindGroupLayoutEntry(4, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(Config)), //cubemap
		BindGroupLayoutEntry(4, WGPUShaderStage_Compute, WGPUTextureFormat_RGBA8Unorm, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(5, WGPUShaderStage_Compute, WGPUTextureViewDimension_2D, WGPUTextureSampleType_UnfilterableFloat),
	};
	
	computeTexturesBindGroupAlpha = new BindGroup(device, textureLayoutEntries);

	computeTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(0, rayOriginTextureAlpha->getTextureView()));
	computeTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(1, rayDirectionTextureAlpha->getTextureView()));
	computeTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(2, rayOriginTextureBeta->getTextureView()));
	computeTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(3, rayDirectionTextureBeta->getTextureView()));
	computeTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(4, colorTextureAlpha->getTextureView()));
	computeTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(5, colorTextureBeta->getTextureView()));

	computeTexturesBindGroupAlpha->finalize();

	computeTexturesBindGroupBeta = new BindGroup(device, textureLayoutEntries);

	computeTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(0, rayOriginTextureBeta->getTextureView()));
	computeTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(1, rayDirectionTextureBeta->getTextureView()));
	computeTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(2, rayOriginTextureAlpha->getTextureView()));
	computeTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(3, rayDirectionTextureAlpha->getTextureView()));
	computeTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(4, colorTextureBeta->getTextureView()));
	computeTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(5, colorTextureAlpha->getTextureView()));

	computeTexturesBindGroupBeta->finalize();



	//BVH+spheres
	initBVHDataBindGroup();

	//pathtracing pipeline
	WGPUComputePipelineDescriptor pathtracingPipelineDesc = {};
	pathtracingPipelineDesc.nextInChain = nullptr;

	pathtracingPipelineDesc.compute.module = shaderModule;
	pathtracingPipelineDesc.compute.entryPoint = "main";

	
	std::vector<WGPUBindGroupLayout> pathtracingBindGroupLayouts(4);
	pathtracingBindGroupLayouts[0] = pathtracingDataBindGroup->getLayout();
	pathtracingBindGroupLayouts[1] = computeTexturesBindGroupAlpha->getLayout();
	pathtracingBindGroupLayouts[2] = scene->getSkyboxesContainer()->getSkyboxBindGroup("Sky")->getLayout();
	//pathtracingBindGroupLayouts[3] = debugCounterBindGroupAlpha->getLayout();
	pathtracingBindGroupLayouts[3] = bvhDataBindGroup->getLayout();
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 4;
	layoutDesc.bindGroupLayouts = pathtracingBindGroupLayouts.data();



	WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
	pathtracingPipelineDesc.layout = layout;

	pathtracingPipeline = wgpuDeviceCreateComputePipeline(device, &pathtracingPipelineDesc);
	std::cout << "pathtracingPipeline compute pipeline: " << pathtracingPipeline << std::endl;

}
void Renderer::initDebugPipeline(){
	
	WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("resources\\Shaders\\debug_shader.wgsl", device);


	colorTextureDebug = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA8Unorm);

	//data group
	std::vector<BindGroupLayoutEntry> dataLayoutEntries = {
		//BindGroupLayoutEntry(0,WGPUShaderStage_Compute,WGPUBufferBindingType_ReadOnlyStorage,sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size()),
		//BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUBufferBindingType_ReadOnlyStorage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size()),
		BindGroupLayoutEntry(0, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(CameraUBO)),
		//BindGroupLayoutEntry(3, WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,sizeof(DebugData)),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,32 * sizeof(DebugData)),
		//BindGroupLayoutEntry(5, WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,sizeof(uint32_t)),
		BindGroupLayoutEntry(2, WGPUShaderStage_Compute, WGPUTextureFormat_RGBA8Unorm, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(3, WGPUShaderStage_Compute,WGPUBufferBindingType_Uniform,sizeof(Config)),
		BindGroupLayoutEntry(4, WGPUShaderStage_Compute,WGPUBufferBindingType_Uniform,sizeof(DebugConfig)),
	};

	debugDataBindGroup = new BindGroup(device, dataLayoutEntries);

	//debugDataBindGroup->addEntry(new BufferBindGroupEntry(0, bvhBuffer->getBuffer(), 0, bvhBuffer->getSize()));
	//debugDataBindGroup->addEntry(new BufferBindGroupEntry(1, spheresBuffer->getBuffer(), 0, spheresBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(0, cameraBuffer->getBuffer(), 0, cameraBuffer->getSize()));
	//debugDataBindGroup->addEntry(new BufferBindGroupEntry(3, debugLineBuffer->getBuffer(), 0, debugLineBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(1, debugLineArrayBuffer->getBuffer(), 0, debugLineArrayBuffer->getSize()));
	//debugDataBindGroup->addEntry(new BufferBindGroupEntry(5, debugIndexAtomicBuffer->getBuffer(), 0, debugIndexAtomicBuffer->getSize()));
	debugDataBindGroup->addEntry(new TextureBindGroupEntry(2, colorTextureDebug->getTextureView()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(3, configBuffer->getBuffer(), 0, configBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(4, debugConfigBuffer->getBuffer(), 0, debugConfigBuffer->getSize()));
	debugDataBindGroup->finalize();


	WGPUComputePipelineDescriptor pathtracingPipelineDesc = {};
	pathtracingPipelineDesc.nextInChain = nullptr;

	pathtracingPipelineDesc.compute.module = shaderModule;
	pathtracingPipelineDesc.compute.entryPoint = "main";

	
	std::vector<WGPUBindGroupLayout> pathtracingBindGroupLayouts(2);
	pathtracingBindGroupLayouts[0] = debugDataBindGroup->getLayout();
	pathtracingBindGroupLayouts[1] = bvhDataBindGroup->getLayout();
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 2;
	layoutDesc.bindGroupLayouts = pathtracingBindGroupLayouts.data();



	WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
	pathtracingPipelineDesc.layout = layout;

	debugComputePipeline = wgpuDeviceCreateComputePipeline(device, &pathtracingPipelineDesc);
	std::cout << "debugComputePipeline compute pipeline: " << debugComputePipeline << std::endl;
	//screen 

	std::vector<BindGroupLayoutEntry> screenLayoutEntries = {
		BindGroupLayoutEntry(0, WGPUShaderStage_Fragment, WGPUTextureViewDimension_2D, WGPUTextureSampleType_UnfilterableFloat),
	};

	debugScreenBindGroup = new BindGroup(device, screenLayoutEntries);

	debugScreenBindGroup->addEntry(new TextureBindGroupEntry(0, colorTextureDebug->getTextureView()));

	debugScreenBindGroup->finalize();



	shaderModule = ShaderLoader::loadShaderModule("resources\\Shaders\\debug_screen_shader.wgsl", device);
	//pipeline
	WGPURenderPipelineDescriptor screenPipelineDesc = {};
	screenPipelineDesc.nextInChain = nullptr;

	screenPipelineDesc.vertex.module = shaderModule;
	screenPipelineDesc.vertex.entryPoint = "vs_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	screenPipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
	fragmentState.entryPoint = "fs_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	screenPipelineDesc.multisample.mask = ~0u;
	screenPipelineDesc.multisample.alphaToCoverageEnabled = false;
	screenPipelineDesc.multisample.count = 1;

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
	//colorTarget.blend = &blendState;
	colorTarget.writeMask = WGPUColorWriteMask_All;

	fragmentState.targetCount = 1;
	fragmentState.targets = &colorTarget;

	screenPipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	screenPipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
	screenPipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
	screenPipelineDesc.primitive.cullMode = WGPUCullMode_None;


	std::vector<WGPUBindGroupLayout> screenBindGroupLayouts(1);
	screenBindGroupLayouts[0] = debugScreenBindGroup->getLayout();
	
	WGPUPipelineLayoutDescriptor screenLayoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	screenLayoutDesc.bindGroupLayoutCount = 1;
	screenLayoutDesc.bindGroupLayouts = screenBindGroupLayouts.data();

	WGPUPipelineLayout screenLayout = wgpuDeviceCreatePipelineLayout(device, &screenLayoutDesc);
	screenPipelineDesc.layout = screenLayout;

	debugScreenPipeline = wgpuDeviceCreateRenderPipeline(device, &screenPipelineDesc);
	
}

void Renderer::initAccumulationPipeline() {
	WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("resources\\Shaders\\screen_shader_accumulated.wgsl", device);

	//data group
	std::vector<BindGroupLayoutEntry> configLayoutEntries = {
		BindGroupLayoutEntry(0, WGPUShaderStage_Fragment, WGPUSamplerBindingType_Filtering), //sampler
		BindGroupLayoutEntry(1, WGPUShaderStage_Fragment, WGPUBufferBindingType_Uniform, sizeof(Config)),
		BindGroupLayoutEntry(2, WGPUShaderStage_Fragment, WGPUBufferBindingType_Uniform, sizeof(BilateralFilterConfig)),
	};

	configBindGroup = new BindGroup(device, configLayoutEntries);

	configBindGroup->addEntry(new SamplerBindGroupEntry(0, colorTextureAlpha->getSampler()));
	configBindGroup->addEntry(new BufferBindGroupEntry(1, configBuffer->getBuffer(), 0, configBuffer->getSize()));
	configBindGroup->addEntry(new BufferBindGroupEntry(2, bilateralFilterConfigBuffer->getBuffer(), 0, bilateralFilterConfigBuffer->getSize()));
	configBindGroup->finalize();

	//sample group
	std::vector<BindGroupLayoutEntry> sampleLayoutEntries = {
		BindGroupLayoutEntry(0, WGPUShaderStage_Fragment, WGPUTextureViewDimension_2D, WGPUTextureSampleType_UnfilterableFloat), //sampler
	};

	sampleBindGroupAlpha = new BindGroup(device, sampleLayoutEntries);

	sampleBindGroupAlpha->addEntry(new TextureBindGroupEntry(0, colorTextureAlpha->getTextureView()));

	sampleBindGroupAlpha->finalize();

	sampleBindGroupBeta = new BindGroup(device, sampleLayoutEntries);

	sampleBindGroupBeta->addEntry(new TextureBindGroupEntry(0, colorTextureBeta->getTextureView()));

	sampleBindGroupBeta->finalize();

	//screen accumulation group
	std::vector<BindGroupLayoutEntry> accumulationLayoutEntries = {
		BindGroupLayoutEntry(0, WGPUShaderStage_Fragment, WGPUTextureFormat_RGBA8Unorm, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(1, WGPUShaderStage_Fragment, WGPUTextureViewDimension_2D, WGPUTextureSampleType_UnfilterableFloat),
	};

	//TODO: new textures
	Texture* accumulationTextureAlpha = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA8Unorm);
	Texture* accumulationTextureBeta = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA8Unorm);


	accumulationTexturesBindGroupAlpha = new BindGroup(device, accumulationLayoutEntries);
	accumulationTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(0, accumulationTextureAlpha->getTextureView()));
	accumulationTexturesBindGroupAlpha->addEntry(new TextureBindGroupEntry(1, accumulationTextureBeta->getTextureView()));
	accumulationTexturesBindGroupAlpha->finalize();

	accumulationTexturesBindGroupBeta = new BindGroup(device, accumulationLayoutEntries);
	accumulationTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(0, accumulationTextureBeta->getTextureView()));
	accumulationTexturesBindGroupBeta->addEntry(new TextureBindGroupEntry(1, accumulationTextureAlpha->getTextureView()));
	accumulationTexturesBindGroupBeta->finalize();

	//pipeline
	WGPURenderPipelineDescriptor screenPipelineDesc = {};
	screenPipelineDesc.nextInChain = nullptr;

	screenPipelineDesc.vertex.module = shaderModule;
	screenPipelineDesc.vertex.entryPoint = "vs_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	screenPipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
	fragmentState.entryPoint = "fs_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	screenPipelineDesc.multisample.mask = ~0u;
	screenPipelineDesc.multisample.alphaToCoverageEnabled = false;
	screenPipelineDesc.multisample.count = 1;

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
	//colorTarget.blend = &blendState;
	colorTarget.writeMask = WGPUColorWriteMask_All;

	fragmentState.targetCount = 1;
	fragmentState.targets = &colorTarget;

	screenPipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	screenPipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
	screenPipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
	screenPipelineDesc.primitive.cullMode = WGPUCullMode_None;


	std::vector<WGPUBindGroupLayout> screenBindGroupLayouts(3);
	screenBindGroupLayouts[0] = configBindGroup->getLayout();
	screenBindGroupLayouts[1] = sampleBindGroupAlpha->getLayout();
	screenBindGroupLayouts[2] = accumulationTexturesBindGroupAlpha->getLayout();
	// Pipeline layout
	WGPUPipelineLayoutDescriptor screenLayoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	screenLayoutDesc.bindGroupLayoutCount = 3;
	screenLayoutDesc.bindGroupLayouts = screenBindGroupLayouts.data();

	WGPUPipelineLayout screenLayout = wgpuDeviceCreatePipelineLayout(device, &screenLayoutDesc);
	screenPipelineDesc.layout = screenLayout;

	accumulationPipeline = wgpuDeviceCreateRenderPipeline(device, &screenPipelineDesc);
}

void Renderer::initMaterialSets() {
	MaterialSet metals;
	metals.addMaterial(Material(glm::vec4(0.944f, 0.776f, 0.373f, 0.08f)));
	metals.addMaterial(Material(glm::vec4(0.95f, 0.64f, 0.54f, 0.13f)));
	metals.addMaterial(Material(glm::vec4(0.912f, 0.914f, 0.920f, 0.1f)));
	metals.addMaterial(Material(glm::vec4(0.531f, 0.512f, 0.496f,0.02)));
	materialSets.push_back(metals);

	MaterialSet diffuse;
	diffuse.addMaterial(Material(glm::vec4(0.830f, 0.791f, 0.753f, .4f)));
	diffuse.addMaterial(Material(glm::vec4(0.440f, 0.386f, 0.231f, .8f)));
	diffuse.addMaterial(Material(glm::vec4(0.713, 0.170, 0.026f, 0.6f)));
	diffuse.addMaterial(Material(glm::vec4(0.634f, 0.532f, 0.111f, 0.7f)));
	materialSets.push_back(diffuse);

	MaterialSet diffuseLights;
	diffuseLights.addMaterial(Material(glm::vec4(0.502f, 0.502f, 0.502f, .8f)));
	diffuseLights.addMaterial(Material(glm::vec4(0.784f, 0.196f, 0.196f, .5f)));
	diffuseLights.addMaterial(Material(glm::vec4(0.196f, 0.392f, 0.784f, 0.6f)));
	diffuseLights.addMaterial(Material(glm::vec4(1.0f, 0.85f, 0.6f, 0.0f)));
	materialSets.push_back(diffuseLights);

	MaterialSet glass;
	glass.addMaterial(Material(glm::vec4(1.f, 1.f, 1.f, 2.0f)));
	glass.addMaterial(Material(glm::vec4(0.6f, 0.8f, 0.6f, 2.0f)));
	glass.addMaterial(Material(glm::vec4(0.8f, 0.6f, 0.3f, 2.0f)));
	glass.addMaterial(Material(glm::vec4(0.4f, 0.6f, 0.8f, 2.0f)));

	materialSets.push_back(glass);

	MaterialSet mix;
	mix.addMaterial(Material(glm::vec4(0.944f, 0.776f, 0.373f, 0.031f)));
	mix.addMaterial(Material(glm::vec4(0.440f, 0.386f, 0.231f, .8f)));
	mix.addMaterial(Material(glm::vec4(1.f, 1.f, 1.f, 2.0f)));
	mix.addMaterial(Material(glm::vec4(1.f, 1.f, 1.f, 0.f)));
	materialSets.push_back(mix);
}

void Renderer::loadNewSpheres() {
	if (scene->isBVHChanged()) {
		//if (bvhBuffer != NULL) bvhBuffer->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0, sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size());
		//if (spheresBuffer != NULL) spheresBuffer->write(queue, scene->getBVH()->getSpheresGPU().data(), 0, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size());
		
		scene->SetBVHChanged(false);
		initBVHDataBindGroup();
		//initPathtracingPipeline();
	}
}

void Renderer::initBVHDataBindGroup() {
	if (bufferBVH != nullptr) {
		wgpuBufferRelease(bufferBVH->getBuffer());
		delete bufferBVH;

	}
	if (bufferSpheres != nullptr) {
		wgpuBufferRelease(bufferSpheres->getBuffer());
		delete bufferSpheres;
	}

	bufferBVH = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size());
	bufferBVH->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0);
	bufferSpheres = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size());
	bufferSpheres->write(queue, scene->getBVH()->getSpheresGPU().data(), 0);

	//BVH+spheres
	std::vector<BindGroupLayoutEntry> bvhLayoutEntries = {
		//(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUTextureFormat format, WGPUStorageTextureAccess access)
		BindGroupLayoutEntry(0,WGPUShaderStage_Compute,WGPUBufferBindingType_ReadOnlyStorage,sizeof(BVHNodeSSBO)),// * scene->getBVH()->getBVHSSBOs().size()),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUBufferBindingType_ReadOnlyStorage, sizeof(SphereGPU)),// * scene->getBVH()->getSpheresGPU().size()),
	};



	bvhDataBindGroup = new BindGroup(device, bvhLayoutEntries);
	bvhDataBindGroup->addEntry(new BufferBindGroupEntry(0, bufferBVH->getBuffer(), 0, bufferBVH->getSize()));
	bvhDataBindGroup->addEntry(new BufferBindGroupEntry(1, bufferSpheres->getBuffer(), 0, bufferSpheres->getSize()));
	bvhDataBindGroup->finalize();
}