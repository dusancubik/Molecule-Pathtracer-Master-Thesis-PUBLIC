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
    
	initPathtracingPipeline();
	initDebugPipeline();
	initAccumulationPipeline();
}

void Renderer::render(WGPUTextureView& nextTexture) {
	if (scene->isBVHChanged()) {
		if(bvhBuffer != NULL) bvhBuffer->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0);
		if(spheresBuffer != NULL) spheresBuffer->write(queue, scene->getBVH()->getSpheresGPU().data(), 0);
		scene->SetBVHChanged(false);
	}

	if (scene->getCamera()->didMove()) {
		sampleId = 0; iteration = 0; bilateralFilterConfig->accumulationFinished = 0;
	}
	scene->getCamera()->cameraUBO.position.w = iteration / 1.0f;
	
	config->currentIteration = iteration;
	config->currentSample = sampleId;
	config->time = glfwGetTime();
	float time = 0.f;
	config->uniformRandom = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (1.f)));
	configBuffer->write(queue, config, 0);
	debugConfigBuffer->write(queue, debugConfig, 0);
	
	//wgpuQueueWriteBuffer(queue, configBuffer, 0, &config, sizeof(Config));
	cameraBuffer->write(queue, scene->getCamera()->getCameraUbo(), 0);

	if (iteration < config->maxIterations - 1) {
		iteration++;
	}
	else if (iteration >= config->maxIterations - 1 && sampleId < config->maxSamples - 1) {
		iteration = 0;
		sampleId++;
	}
	else if (sampleId == config->maxSamples-1) {
		//accumulation finished
		bilateralFilterConfig->accumulationFinished = 1;
	}
	bilateralFilterConfigBuffer->write(queue, bilateralFilterConfig, 0);

	if (debugMode) {
		render_debug(nextTexture);
		return;
	}
	//if (iteration % 5 == 0) iteration = 0;
	//std::cout << "config.currentSample: "<< config.currentSample << std::endl;
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
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 1, ((iteration % 2 == 0) || iteration == 0)  ? computeTexturesBindGroupAlpha->getBindGroup() : computeTexturesBindGroupBeta->getBindGroup(), 0, nullptr);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 2, scene->getSkyboxesContainer()->getCurrentSkyboxBindGroup()->getBindGroup(), 0, nullptr);
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 3, ((sampleId % 2 == 0)) ? debugCounterBindGroupAlpha->getBindGroup() : debugCounterBindGroupBeta->getBindGroup(), 0, nullptr);
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

	
	/*if (scene->isBVHChanged()) {
		if (bvhBuffer != NULL) bvhBuffer->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0);
		if (spheresBuffer != NULL) spheresBuffer->write(queue, scene->getBVH()->getSpheresGPU().data(), 0);
		scene->SetBVHChanged(false);
	}

	if (scene->getCamera()->didMove()) {
		sampleId = 0; iteration = 0;
	}
	scene->getCamera()->cameraUBO.position.w = iteration / 1.0f;

	config->currentIteration = iteration;
	config->currentSample = sampleId;
	config->time = glfwGetTime();
	float time = 0.f;
	config->uniformRandom = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (1.f)));;
	configBuffer->write(queue, config, 0);
	bilateralFilterConfigBuffer->write(queue, bilateralFilterConfig, 0);
	//wgpuQueueWriteBuffer(queue, configBuffer, 0, &config, sizeof(Config));
	cameraBuffer->write(queue, scene->getCamera()->getCameraUbo(), 0);

	if (iteration < config->maxIterations - 1) {
		iteration++;
	}
	else if (iteration >= config->maxIterations - 1 && sampleId < config->maxSamples - 1) {
		iteration = 0;
		sampleId++;
	}

	//if (iteration % 5 == 0) iteration = 0;
	//std::cout << "config.currentSample: "<< config.currentSample << std::endl;
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
	renderPassColorAttachment.clearValue = WGPUColor{ 0.05, 0.05, 0.75, 1.0 };
	renderPassDesc.colorAttachmentCount = 1;
	renderPassDesc.colorAttachments = &renderPassColorAttachment;

	WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);



	wgpuRenderPassEncoderSetPipeline(renderPass, debugPipeline);

	wgpuRenderPassEncoderSetBindGroup(renderPass, 0, debugDataBindGroup->getBindGroup(), 0, nullptr);

	//else wgpuRenderPassEncoderSetBindGroup(renderPass, 1, screenDataTextureBindGroup1, 0, nullptr);
	wgpuRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);

	wgpuRenderPassEncoderEnd(renderPass);

	WGPUCommandBufferDescriptor cmdBufferDesc = {};
	cmdBufferDesc.nextInChain = nullptr;
	cmdBufferDesc.label = "Command buffer";
	WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
	wgpuQueueSubmit(queue, 1, &commandBuffer);*/
}

void Renderer::initPathtracingPipeline(){
	WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Shaders\\raytracing_kernel_bvh_accumulated.wgsl", device);

	

	//pathtracing data group
	std::vector<BindGroupLayoutEntry> layoutEntries = {
		//(uint32_t binding, WGPUShaderStageFlags stageFlags, WGPUTextureFormat format, WGPUStorageTextureAccess access)
		BindGroupLayoutEntry(0,WGPUShaderStage_Compute,WGPUBufferBindingType_ReadOnlyStorage,sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size()),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUBufferBindingType_ReadOnlyStorage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size()),
		BindGroupLayoutEntry(2, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(CameraUBO)),
		BindGroupLayoutEntry(3, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(Config)),
		//BindGroupLayoutEntry(4, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(Config)), //cubemap
		//BindGroupLayoutEntry(4, WGPUShaderStage_Compute, WGPUTextureViewDimension_Cube, WGPUTextureSampleType_Float),
		//BindGroupLayoutEntry(5, WGPUShaderStage_Compute, WGPUSamplerBindingType_Filtering) //sampler
	};

	bvhBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size());
	bvhBuffer->write(queue, scene->getBVH()->getBVHSSBOs().data(), 0);
	spheresBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size());
	spheresBuffer->write(queue, scene->getBVH()->getSpheresGPU().data(), 0);
	//Here: camera buffer 
	configBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(Config));
	configBuffer->write(queue,config,0);
	bilateralFilterConfigBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(BilateralFilterConfig));
	bilateralFilterConfigBuffer->write(queue, bilateralFilterConfig, 0);
	Texture* skyboxTexture = new Texture(device, 0, 0, WGPUTextureFormat_RGBA8Unorm, scene->getSkybox()->getCubemapData());//sampler inside

	pathtracingDataBindGroup = new BindGroup(device,layoutEntries);

	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(0, bvhBuffer->getBuffer(), 0, bvhBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(1, spheresBuffer->getBuffer(), 0, spheresBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(2, cameraBuffer->getBuffer(), 0, cameraBuffer->getSize()));
	pathtracingDataBindGroup->addEntry(new BufferBindGroupEntry(3, configBuffer->getBuffer(), 0, configBuffer->getSize()));
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

	//DEBUG COUNTER
	Texture* debugCounterTextureAlpha = new Texture(device, 1280, 720, WGPUTextureFormat_R32Uint);
	Texture* debugCounterTextureBeta = new Texture(device, 1280, 720, WGPUTextureFormat_R32Uint);

	

	debugLineBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(DebugData));
	debugLineArrayBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, 128*sizeof(DebugData));
	debugIndexAtomicBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(uint32_t));
	debugConfigBuffer = new Buffer(device, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(DebugConfig));

	std::vector<BindGroupLayoutEntry> debugCounterLayoutEntries = {
		BindGroupLayoutEntry(0, WGPUShaderStage_Compute, WGPUTextureFormat_R32Uint, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUTextureViewDimension_2D, WGPUTextureSampleType_Uint),
		BindGroupLayoutEntry(2,WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,sizeof(DebugData)),
		BindGroupLayoutEntry(3,WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,128*sizeof(DebugData)),
		BindGroupLayoutEntry(4,WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,sizeof(uint32_t)),
		BindGroupLayoutEntry(5, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(DebugConfig)),
	};
	debugCounterBindGroupAlpha = new BindGroup(device, debugCounterLayoutEntries);
	
	debugCounterBindGroupAlpha->addEntry(new TextureBindGroupEntry(0, debugCounterTextureAlpha->getTextureView()));
	debugCounterBindGroupAlpha->addEntry(new TextureBindGroupEntry(1, debugCounterTextureBeta->getTextureView()));
	debugCounterBindGroupAlpha->addEntry(new BufferBindGroupEntry(2, debugLineBuffer->getBuffer(), 0, debugLineBuffer->getSize()));
	debugCounterBindGroupAlpha->addEntry(new BufferBindGroupEntry(3, debugLineArrayBuffer->getBuffer(), 0, debugLineArrayBuffer->getSize()));
	debugCounterBindGroupAlpha->addEntry(new BufferBindGroupEntry(4, debugIndexAtomicBuffer->getBuffer(), 0, debugIndexAtomicBuffer->getSize()));
	debugCounterBindGroupAlpha->addEntry(new BufferBindGroupEntry(5, debugConfigBuffer->getBuffer(), 0, debugConfigBuffer->getSize()));
	debugCounterBindGroupAlpha->finalize();

	debugCounterBindGroupBeta = new BindGroup(device, debugCounterLayoutEntries);

	debugCounterBindGroupBeta->addEntry(new TextureBindGroupEntry(0, debugCounterTextureBeta->getTextureView()));
	debugCounterBindGroupBeta->addEntry(new TextureBindGroupEntry(1, debugCounterTextureAlpha->getTextureView()));
	debugCounterBindGroupBeta->addEntry(new BufferBindGroupEntry(2, debugLineBuffer->getBuffer(), 0, debugLineBuffer->getSize()));
	debugCounterBindGroupBeta->addEntry(new BufferBindGroupEntry(3, debugLineArrayBuffer->getBuffer(), 0, debugLineArrayBuffer->getSize()));
	debugCounterBindGroupBeta->addEntry(new BufferBindGroupEntry(4, debugIndexAtomicBuffer->getBuffer(), 0, debugIndexAtomicBuffer->getSize()));
	debugCounterBindGroupBeta->addEntry(new BufferBindGroupEntry(5, debugConfigBuffer->getBuffer(), 0, debugConfigBuffer->getSize()));
	debugCounterBindGroupBeta->finalize();
	//pathtracing pipeline
	WGPUComputePipelineDescriptor pathtracingPipelineDesc = {};
	pathtracingPipelineDesc.nextInChain = nullptr;

	pathtracingPipelineDesc.compute.module = shaderModule;
	pathtracingPipelineDesc.compute.entryPoint = "main";

	// Pipeline layout
	std::vector<WGPUBindGroupLayout> pathtracingBindGroupLayouts(4);
	pathtracingBindGroupLayouts[0] = pathtracingDataBindGroup->getLayout();
	pathtracingBindGroupLayouts[1] = computeTexturesBindGroupAlpha->getLayout();
	pathtracingBindGroupLayouts[2] = scene->getSkyboxesContainer()->getSkyboxBindGroup("Default")->getLayout();
	pathtracingBindGroupLayouts[3] = debugCounterBindGroupAlpha->getLayout();
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
	/*WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Shaders\\debug_shader.wgsl", device);

	//data group
	std::vector<BindGroupLayoutEntry> dataLayoutEntries = {
		BindGroupLayoutEntry(0,WGPUShaderStage_Fragment,WGPUBufferBindingType_ReadOnlyStorage,sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size()),
		BindGroupLayoutEntry(1, WGPUShaderStage_Fragment, WGPUBufferBindingType_ReadOnlyStorage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size()),
		BindGroupLayoutEntry(2, WGPUShaderStage_Fragment, WGPUBufferBindingType_Uniform, sizeof(CameraUBO)),
		BindGroupLayoutEntry(3, WGPUShaderStage_Fragment,WGPUBufferBindingType_Storage,sizeof(DebugData)),
		BindGroupLayoutEntry(4, WGPUShaderStage_Fragment,WGPUBufferBindingType_Storage,32*sizeof(DebugData)),
		BindGroupLayoutEntry(5, WGPUShaderStage_Fragment,WGPUBufferBindingType_Storage,sizeof(uint32_t)),
	};

	debugDataBindGroup = new BindGroup(device, dataLayoutEntries);

	debugDataBindGroup->addEntry(new BufferBindGroupEntry(0, bvhBuffer->getBuffer(), 0, bvhBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(1, spheresBuffer->getBuffer(), 0, spheresBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(2, cameraBuffer->getBuffer(), 0, cameraBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(3, debugLineBuffer->getBuffer(), 0, debugLineBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(4, debugLineArrayBuffer->getBuffer(), 0, debugLineArrayBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(5, debugIndexAtomicBuffer->getBuffer(), 0, debugIndexAtomicBuffer->getSize()));
	debugDataBindGroup->finalize();

	

	//pipeline
	WGPURenderPipelineDescriptor debugPipelineDesc = {};
	debugPipelineDesc.nextInChain = nullptr;

	debugPipelineDesc.vertex.module = shaderModule;
	debugPipelineDesc.vertex.entryPoint = "vs_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	debugPipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
	fragmentState.entryPoint = "fs_main";
	fragmentState.constantCount = 0;
	fragmentState.constants = nullptr;

	debugPipelineDesc.multisample.mask = ~0u;
	debugPipelineDesc.multisample.alphaToCoverageEnabled = false;
	debugPipelineDesc.multisample.count = 1;

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

	debugPipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	debugPipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
	debugPipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
	debugPipelineDesc.primitive.cullMode = WGPUCullMode_None;


	std::vector<WGPUBindGroupLayout> debugBindGroupLayouts(1);
	debugBindGroupLayouts[0] = debugDataBindGroup->getLayout();
	// Pipeline layout
	WGPUPipelineLayoutDescriptor screenLayoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	screenLayoutDesc.bindGroupLayoutCount = 1;
	screenLayoutDesc.bindGroupLayouts = debugBindGroupLayouts.data();

	WGPUPipelineLayout screenLayout = wgpuDeviceCreatePipelineLayout(device, &screenLayoutDesc);
	debugPipelineDesc.layout = screenLayout;

	debugPipeline = wgpuDeviceCreateRenderPipeline(device, &debugPipelineDesc);*/
	WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Shaders\\debug_shader.wgsl", device);

	colorTextureDebug = new Texture(device, 1280, 720, WGPUTextureFormat_RGBA8Unorm);

	//data group
	std::vector<BindGroupLayoutEntry> dataLayoutEntries = {
		BindGroupLayoutEntry(0,WGPUShaderStage_Compute,WGPUBufferBindingType_ReadOnlyStorage,sizeof(BVHNodeSSBO) * scene->getBVH()->getBVHSSBOs().size()),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUBufferBindingType_ReadOnlyStorage, sizeof(SphereGPU) * scene->getBVH()->getSpheresGPU().size()),
		BindGroupLayoutEntry(2, WGPUShaderStage_Compute, WGPUBufferBindingType_Uniform, sizeof(CameraUBO)),
		BindGroupLayoutEntry(3, WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,sizeof(DebugData)),
		BindGroupLayoutEntry(4, WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,32 * sizeof(DebugData)),
		BindGroupLayoutEntry(5, WGPUShaderStage_Compute,WGPUBufferBindingType_Storage,sizeof(uint32_t)),
		BindGroupLayoutEntry(6, WGPUShaderStage_Compute, WGPUTextureFormat_RGBA8Unorm, WGPUStorageTextureAccess_WriteOnly),
		BindGroupLayoutEntry(7, WGPUShaderStage_Compute,WGPUBufferBindingType_Uniform,sizeof(Config)),
		BindGroupLayoutEntry(8, WGPUShaderStage_Compute,WGPUBufferBindingType_Uniform,sizeof(DebugConfig))
	};

	debugDataBindGroup = new BindGroup(device, dataLayoutEntries);

	debugDataBindGroup->addEntry(new BufferBindGroupEntry(0, bvhBuffer->getBuffer(), 0, bvhBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(1, spheresBuffer->getBuffer(), 0, spheresBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(2, cameraBuffer->getBuffer(), 0, cameraBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(3, debugLineBuffer->getBuffer(), 0, debugLineBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(4, debugLineArrayBuffer->getBuffer(), 0, debugLineArrayBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(5, debugIndexAtomicBuffer->getBuffer(), 0, debugIndexAtomicBuffer->getSize()));
	debugDataBindGroup->addEntry(new TextureBindGroupEntry(6, colorTextureDebug->getTextureView()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(7, configBuffer->getBuffer(), 0, configBuffer->getSize()));
	debugDataBindGroup->addEntry(new BufferBindGroupEntry(8, debugConfigBuffer->getBuffer(), 0, debugConfigBuffer->getSize()));
	debugDataBindGroup->finalize();


	WGPUComputePipelineDescriptor pathtracingPipelineDesc = {};
	pathtracingPipelineDesc.nextInChain = nullptr;

	pathtracingPipelineDesc.compute.module = shaderModule;
	pathtracingPipelineDesc.compute.entryPoint = "main";

	// Pipeline layout
	std::vector<WGPUBindGroupLayout> pathtracingBindGroupLayouts(1);
	pathtracingBindGroupLayouts[0] = debugDataBindGroup->getLayout();
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 1;
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



	shaderModule = ShaderLoader::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Shaders\\debug_screen_shader.wgsl", device);
	//pipeline
	WGPURenderPipelineDescriptor screenPipelineDesc = {};
	screenPipelineDesc.nextInChain = nullptr;

	screenPipelineDesc.vertex.module = shaderModule;
	screenPipelineDesc.vertex.entryPoint = "vert_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	screenPipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
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


	std::vector<WGPUBindGroupLayout> screenBindGroupLayouts(1);
	screenBindGroupLayouts[0] = debugScreenBindGroup->getLayout();
	// Pipeline layout
	WGPUPipelineLayoutDescriptor screenLayoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	screenLayoutDesc.bindGroupLayoutCount = 1;
	screenLayoutDesc.bindGroupLayouts = screenBindGroupLayouts.data();

	WGPUPipelineLayout screenLayout = wgpuDeviceCreatePipelineLayout(device, &screenLayoutDesc);
	screenPipelineDesc.layout = screenLayout;

	debugScreenPipeline = wgpuDeviceCreateRenderPipeline(device, &screenPipelineDesc);
	
}

void Renderer::initAccumulationPipeline() {
	WGPUShaderModule shaderModule = ShaderLoader::loadShaderModule("E:\\MUNI\\Diplomka\\dusancubik-master-thesis\\apps\\medvis\\resources\\Shaders\\screen_shader_accumulated.wgsl", device);

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
	screenPipelineDesc.vertex.entryPoint = "vert_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	screenPipelineDesc.fragment = &fragmentState;
	fragmentState.module = shaderModule;
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
