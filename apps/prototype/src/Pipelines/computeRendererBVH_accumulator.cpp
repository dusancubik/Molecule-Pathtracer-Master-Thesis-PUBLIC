#include "computeRenderer.hpp"
#include "../KdTree/kdTreeRopes.hpp"

bool ComputeRendererBVHAccumulator::initCamera() {
	camera = std::make_shared<PROTO_Camera>(1280, 720, glm::vec3(0.f, 0.f, 3.0f));
	return true;
}



void ComputeRendererBVHAccumulator::initRaytraycingDataTexturesBindGroups() {
	
	// 
	std::vector<WGPUBindGroupLayoutEntry> raytracingDataBindingLayout1(6);
	createBindingLayout(0, 0, raytracingDataBindingLayout1[0], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingDataBindingLayout1[0].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
	raytracingDataBindingLayout1[0].storageTexture.format = WGPUTextureFormat_RGBA32Float;
	raytracingDataBindingLayout1[0].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
	raytracingDataBindingLayout1[0].sampler = { 0 };
	createBindingLayout(1, 0, raytracingDataBindingLayout1[1], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingDataBindingLayout1[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
	raytracingDataBindingLayout1[1].storageTexture.format = WGPUTextureFormat_RGBA32Float;
	raytracingDataBindingLayout1[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
	raytracingDataBindingLayout1[1].sampler = { 0 };

	WGPUTextureBindingLayout textureBindingLayout1{};
	textureBindingLayout1.sampleType = WGPUTextureSampleType_UnfilterableFloat;
	createBindingLayout(2, 0, raytracingDataBindingLayout1[2], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingDataBindingLayout1[2].texture = textureBindingLayout1;
	raytracingDataBindingLayout1[2].texture.multisampled = false;
	raytracingDataBindingLayout1[2].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
	raytracingDataBindingLayout1[2].texture.viewDimension = WGPUTextureViewDimension_2D;

	createBindingLayout(3, 0, raytracingDataBindingLayout1[3], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingDataBindingLayout1[3].texture = textureBindingLayout1;
	raytracingDataBindingLayout1[3].texture.multisampled = false;
	raytracingDataBindingLayout1[3].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
	raytracingDataBindingLayout1[3].texture.viewDimension = WGPUTextureViewDimension_2D;

	createBindingLayout(4, 0, raytracingDataBindingLayout1[4], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingDataBindingLayout1[4].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
	raytracingDataBindingLayout1[4].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
	raytracingDataBindingLayout1[4].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
	raytracingDataBindingLayout1[4].sampler = { 0 };

	WGPUTextureBindingLayout textureBindingLayoutColor{};
	textureBindingLayoutColor.sampleType = WGPUTextureSampleType_Float;
	createBindingLayout(5, 0, raytracingDataBindingLayout1[5], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingDataBindingLayout1[5].texture = textureBindingLayoutColor;
	raytracingDataBindingLayout1[5].texture.multisampled = false;
	raytracingDataBindingLayout1[5].texture.sampleType = WGPUTextureSampleType_Float;
	raytracingDataBindingLayout1[5].texture.viewDimension = WGPUTextureViewDimension_2D;

	WGPUBindGroupLayoutDescriptor raytracingBindGroupLayoutDesc = {};
	raytracingBindGroupLayoutDesc.entryCount = 6;
	raytracingBindGroupLayoutDesc.entries = raytracingDataBindingLayout1.data();//&bindingLayout;
	rayTracingDataTexturesBindLayout = wgpuDeviceCreateBindGroupLayout(device, &raytracingBindGroupLayoutDesc);
	

	
	std::vector<WGPUBindGroupEntry> binding(6);
	binding[0].nextInChain = nullptr;
	binding[0].binding = 0;
	binding[0].textureView = origin_buffer_view;

	binding[1].nextInChain = nullptr;
	binding[1].binding = 1;
	binding[1].textureView = direction_buffer_view;

	binding[2].nextInChain = nullptr;
	binding[2].binding = 2;
	binding[2].textureView = origin_buffer_view2;

	binding[3].nextInChain = nullptr;
	binding[3].binding = 3;
	binding[3].textureView = direction_buffer_view2;

	binding[4].nextInChain = nullptr;
	binding[4].binding = 4;
	binding[4].textureView = color_buffer_view;

	binding[5].nextInChain = nullptr;
	binding[5].binding = 5;
	binding[5].textureView = color_buffer_view2;

	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = rayTracingDataTexturesBindLayout;
	
	bindGroupDesc.entryCount = 6;
	bindGroupDesc.entries = binding.data();
	rayTracingDataTexturesBindGroup1 = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

	//2
	std::vector<WGPUBindGroupEntry> binding2(6);
	binding2[0].nextInChain = nullptr;
	binding2[0].binding = 0;
	binding2[0].textureView = origin_buffer_view2;

	binding2[1].nextInChain = nullptr;
	binding2[1].binding = 1;
	binding2[1].textureView = direction_buffer_view2;


	binding2[2].nextInChain = nullptr;
	binding2[2].binding = 2;
	binding2[2].textureView = origin_buffer_view;

	binding2[3].nextInChain = nullptr;
	binding2[3].binding = 3;
	binding2[3].textureView = direction_buffer_view;

	binding2[4].nextInChain = nullptr;
	binding2[4].binding = 4;
	binding2[4].textureView = color_buffer_view2;

	binding2[5].nextInChain = nullptr;
	binding2[5].binding = 5;
	binding2[5].textureView = color_buffer_view;

	WGPUBindGroupDescriptor bindGroupDesc2 = {};
	bindGroupDesc2.nextInChain = nullptr;
	bindGroupDesc2.layout = rayTracingDataTexturesBindLayout;
	
	bindGroupDesc2.entryCount = 6;
	bindGroupDesc2.entries = binding2.data();//&binding;
	rayTracingDataTexturesBindGroup2 = wgpuDeviceCreateBindGroup(device, &bindGroupDesc2);
}

void ComputeRendererBVHAccumulator::initRaytracingBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {
	std::vector<WGPUBindGroupEntry> binding(6);

	//createBindGroupEntry(binding[0], 0, color_buffer_view, 0, ???);


	createBindGroupEntry(binding[0], 0, bvhStorageBuffer, 0, sizeof(BVHNodeSSBO) * bvh->getBVHSSBOs().size());
	createBindGroupEntry(binding[1], 1, spheresStorageBuffer, 0, sizeof(Sphere) * bvh->getSpheres().size());
	createBindGroupEntry(binding[2], 2, uniformBuffer, 0, sizeof(CameraUBO));
	createBindGroupEntry(binding[3], 3, configBuffer, 0, sizeof(Config));

	binding[4].nextInChain = nullptr;
	binding[4].binding = 4;
	binding[4].textureView = cubemapTextureView;

	binding[5].nextInChain = nullptr;

	binding[5].binding = 5;

	binding[5].sampler = cubemapSampler;
	

	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;
	
	bindGroupDesc.entryCount = 6;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}

void ComputeRendererBVHAccumulator::initScreenAccumulationBindGroups() {
	
	std::vector<WGPUBindGroupLayoutEntry> screenDataBindingLayout(2);

	createBindingLayout(0, 0, screenDataBindingLayout[0], WGPUBufferBindingType_Undefined, WGPUShaderStage_Fragment);
	screenDataBindingLayout[0].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
	screenDataBindingLayout[0].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
	screenDataBindingLayout[0].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
	screenDataBindingLayout[0].sampler = { 0 };
	WGPUTextureBindingLayout textureBindingLayout1{};
	textureBindingLayout1.sampleType = WGPUTextureSampleType_Float;
	createBindingLayout(1, 0, screenDataBindingLayout[1], WGPUBufferBindingType_Undefined, WGPUShaderStage_Fragment);
	screenDataBindingLayout[1].texture = textureBindingLayout1;
	screenDataBindingLayout[1].texture.multisampled = false;
	screenDataBindingLayout[1].texture.sampleType = WGPUTextureSampleType_Float;
	screenDataBindingLayout[1].texture.viewDimension = WGPUTextureViewDimension_2D;

	WGPUBindGroupLayoutDescriptor accumulationBindGroupLayoutDesc = {};
	accumulationBindGroupLayoutDesc.entryCount = 2;
	accumulationBindGroupLayoutDesc.entries = screenDataBindingLayout.data();//&bindingLayout;
	screenAccumulationBindLayout = wgpuDeviceCreateBindGroupLayout(device, &accumulationBindGroupLayoutDesc);

	std::vector<WGPUBindGroupEntry> binding(2);
	binding[0].nextInChain = nullptr;
	binding[0].binding = 0;
	binding[0].textureView = accumulation_buffer_view;

	binding[1].nextInChain = nullptr;
	binding[1].binding = 1;
	binding[1].textureView = accumulation_buffer_view2;

	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = screenAccumulationBindLayout;
	
	bindGroupDesc.entryCount = 2;
	bindGroupDesc.entries = binding.data();//&binding;
	screenAccumulationBindGroup1 = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

	
	std::vector<WGPUBindGroupEntry> binding2(2);
	binding2[0].nextInChain = nullptr;
	binding2[0].binding = 0;
	binding2[0].textureView = accumulation_buffer_view2;

	binding2[1].nextInChain = nullptr;
	binding2[1].binding = 1;
	binding2[1].textureView = accumulation_buffer_view;

	WGPUBindGroupDescriptor bindGroupDesc2 = {};
	bindGroupDesc2.nextInChain = nullptr;
	bindGroupDesc2.layout = screenAccumulationBindLayout;
	
	bindGroupDesc2.entryCount = 2;
	bindGroupDesc2.entries = binding2.data();//&binding;
	screenAccumulationBindGroup2 = wgpuDeviceCreateBindGroup(device, &bindGroupDesc2);
}

void ComputeRendererBVHAccumulator::initScreenDataTexturesBindGroups() {
	
	std::vector<WGPUBindGroupLayoutEntry> screenDataBindingLayout(1);

	WGPUTextureBindingLayout textureBindingLayout{};
	textureBindingLayout.sampleType = WGPUTextureSampleType_Float;
	createBindingLayout(0, 0, screenDataBindingLayout[0], WGPUBufferBindingType_Undefined, WGPUShaderStage_Fragment);
	screenDataBindingLayout[0].texture = textureBindingLayout;
	screenDataBindingLayout[0].texture.multisampled = false;
	screenDataBindingLayout[0].texture.sampleType = WGPUTextureSampleType_Float;
	screenDataBindingLayout[0].texture.viewDimension = WGPUTextureViewDimension_2D;
	

	WGPUBindGroupLayoutDescriptor raytracingBindGroupLayoutDesc = {};
	raytracingBindGroupLayoutDesc.entryCount = 1;
	raytracingBindGroupLayoutDesc.entries = screenDataBindingLayout.data();//&bindingLayout;
	screenDataTextureBindLayout = wgpuDeviceCreateBindGroupLayout(device, &raytracingBindGroupLayoutDesc);
	//WGPUBindGroupLayout raytracingDataBindGroupLayout2 = wgpuDeviceCreateBindGroupLayout(device, &raytracingBindGroupLayoutDesc);

	//1
	std::vector<WGPUBindGroupEntry> binding(1);
	binding[0].nextInChain = nullptr;
	binding[0].binding = 0;
	binding[0].textureView = color_buffer_view;

	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = screenDataTextureBindLayout;
	
	bindGroupDesc.entryCount = 1;
	bindGroupDesc.entries = binding.data();//&binding;
	screenDataTextureBindGroup1 = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

	//2
	std::vector<WGPUBindGroupEntry> binding2(1);
	binding2[0].nextInChain = nullptr;
	binding2[0].binding = 0;
	binding2[0].textureView = color_buffer_view2;

	WGPUBindGroupDescriptor bindGroupDesc2 = {};
	bindGroupDesc2.nextInChain = nullptr;
	bindGroupDesc2.layout = screenDataTextureBindLayout;
	
	bindGroupDesc2.entryCount = 1;
	bindGroupDesc2.entries = binding2.data();//&binding;
	screenDataTextureBindGroup2 = wgpuDeviceCreateBindGroup(device, &bindGroupDesc2);
}

void ComputeRendererBVHAccumulator::initScreenBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {
	std::vector<WGPUBindGroupEntry> binding(2);

	//createBindGroupEntry(binding[0], 0, color_buffer_view, 0, ???);

	binding[0].nextInChain = nullptr;

	binding[0].binding = 0;

	binding[0].sampler = sampler;
	createBindGroupEntry(binding[1], 1, configBuffer, 0, sizeof(Config));
	
	WGPUBindGroupDescriptor bindGroupDesc = {};
	bindGroupDesc.nextInChain = nullptr;
	bindGroupDesc.layout = bindGroupLayout;
	
	bindGroupDesc.entryCount = 2;
	bindGroupDesc.entries = binding.data();//&binding;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

	
}

void ComputeRendererBVHAccumulator::initUniforms() {
	//camera
	createBuffer(uniformBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(CameraUBO), camera->getCameraUbo());
	//kd tree
	createBuffer(bvhStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(BVHNodeSSBO) * bvh->getBVHSSBOs().size(), bvh->getBVHSSBOs().data());
	//spheres
	createBuffer(spheresStorageBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, sizeof(Sphere) * bvh->getSpheres().size(), bvh->getSpheres().data());
	
	createBuffer(configBuffer, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, sizeof(Config), &config);
}

void ComputeRendererBVHAccumulator::render(WGPUTextureView &nextTexture) {
	std::cout << "Sample: " << sampleId << " Iteration: " << iteration << "\n";
	//camera->updateCamera();
	if (camera->didMove()) {
		sampleId = 0; iteration = 0;
	}
	camera->cameraUBO.position.w = iteration/1.0f;
	wgpuQueueWriteBuffer(queue, uniformBuffer, 0, camera->getCameraUbo(), sizeof(CameraUBO));
	config.currentIteration = iteration;
	config.currentSample = sampleId;
	config.time = glfwGetTime();
	float time = 0.f;
	config.uniformRandom = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (1.f)));;
	wgpuQueueWriteBuffer(queue, configBuffer, 0, &config, sizeof(Config));
	
	if (iteration < config.maxIterations-1) {
		iteration++;
	}
	else if(iteration >= config.maxIterations-1 && sampleId<config.maxSamples-1){
		iteration = 0;
		sampleId++;
	}
	
	//if (iteration % 5 == 0) iteration = 0;
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
	//ping ponging texture
	wgpuComputePassEncoderSetBindGroup(raytracingPass, 1,(iteration%2)==0? rayTracingDataTexturesBindGroup1: rayTracingDataTexturesBindGroup2, 0, nullptr);
	wgpuComputePassEncoderDispatchWorkgroups(raytracingPass, 1280/8, 720/4, 1);

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
	wgpuRenderPassEncoderSetBindGroup(renderPass, 1, (iteration % 2) == 0? screenDataTextureBindGroup1: screenDataTextureBindGroup2, 0, nullptr);
	wgpuRenderPassEncoderSetBindGroup(renderPass, 2, (config.currentSample % 2) ? screenAccumulationBindGroup1 : screenAccumulationBindGroup2, 0, nullptr);
	
	//else wgpuRenderPassEncoderSetBindGroup(renderPass, 1, screenDataTextureBindGroup1, 0, nullptr);
	wgpuRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);

	wgpuRenderPassEncoderEnd(renderPass);

	WGPUCommandBufferDescriptor cmdBufferDesc = {};
	cmdBufferDesc.nextInChain = nullptr;
	cmdBufferDesc.label = "Command buffer";
	WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
	wgpuQueueSubmit(queue, 1, &commandBuffer);

	
	
}

void ComputeRendererBVHAccumulator::init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) {
	
	RendererBase::init(_spheres, _device, _queue, _swap_chain_default_format);
	//timestamp
	config.currentIteration = 0;
	config.currentSample = 0;

	config.maxIterations = 6;
	config.maxSamples = 1024;
	/*KdTree _kdTree;
	kdTree = &_kdTree;*/
	bvh->construct(spheres);
	//inits
	initCamera();
	initDepthBuffer();
	initUniforms();
	//creating assets
	prepareCubemap();
	//color texture
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
	color_buffer2 = wgpuDeviceCreateTexture(device, &textureDesc);
	WGPUTextureViewDescriptor textureViewDesc{};
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 1;
	color_buffer_view = wgpuTextureCreateView(color_buffer, &textureViewDesc);
	color_buffer_view2 = wgpuTextureCreateView(color_buffer2, &textureViewDesc);

	WGPUSamplerDescriptor samplerDescriptor{};
	samplerDescriptor.addressModeU = WGPUAddressMode_Repeat;
	samplerDescriptor.addressModeV = WGPUAddressMode_Repeat;
	samplerDescriptor.magFilter = WGPUFilterMode_Linear;
	samplerDescriptor.minFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.mipmapFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.maxAnisotropy = 1;

	sampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);

	//origin texture
	WGPUTextureDescriptor textureDescOrigin{};
	textureDesc.nextInChain = nullptr;
	//textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.size = { 1280, 720, 1 };
	textureDesc.format = WGPUTextureFormat_RGBA32Float;
	textureDesc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
	textureDesc.sampleCount = 1;
	textureDesc.mipLevelCount = 1;
	origin_buffer = wgpuDeviceCreateTexture(device, &textureDesc);
	origin_buffer2 = wgpuDeviceCreateTexture(device, &textureDesc);
	WGPUTextureViewDescriptor textureViewDescOrigin{};
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 1;
	origin_buffer_view = wgpuTextureCreateView(origin_buffer, &textureViewDesc);
	origin_buffer_view2 = wgpuTextureCreateView(origin_buffer2, &textureViewDesc);


	//directin texture
	WGPUTextureDescriptor textureDescDirection{};
	textureDescDirection.nextInChain = nullptr;
	//textureDesc.dimension = WGPUTextureDimension_2D;
	textureDescDirection.dimension = WGPUTextureDimension_2D;
	textureDescDirection.size = { 1280, 720, 1 };
	textureDescDirection.format = WGPUTextureFormat_RGBA32Float;
	textureDescDirection.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
	textureDescDirection.sampleCount = 1;
	textureDescDirection.mipLevelCount = 1;
	direction_buffer = wgpuDeviceCreateTexture(device, &textureDescDirection);
	direction_buffer2 = wgpuDeviceCreateTexture(device, &textureDescDirection);
	WGPUTextureViewDescriptor textureViewDescDirection{};
	textureViewDescDirection.mipLevelCount = 1;
	textureViewDescDirection.arrayLayerCount = 1;
	direction_buffer_view = wgpuTextureCreateView(direction_buffer, &textureViewDescDirection);
	direction_buffer_view2 = wgpuTextureCreateView(direction_buffer2, &textureViewDescDirection);
	
	//accumulation texture
	WGPUTextureDescriptor textureDescAccumulation{};
	textureDesc.nextInChain = nullptr;
	//textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.size = { 1280, 720, 1 };
	textureDesc.format = WGPUTextureFormat_RGBA8Unorm;
	textureDesc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
	textureDesc.sampleCount = 1;
	textureDesc.mipLevelCount = 1;
	accumulation_buffer = wgpuDeviceCreateTexture(device, &textureDesc);
	accumulation_buffer2 = wgpuDeviceCreateTexture(device, &textureDesc);
	WGPUTextureViewDescriptor textureViewDescAccumulation{};
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 1;
	accumulation_buffer_view = wgpuTextureCreateView(accumulation_buffer, &textureViewDesc);
	accumulation_buffer_view2 = wgpuTextureCreateView(accumulation_buffer2, &textureViewDesc);

	


	screenShaderModule = ResourceManager::loadShaderModule("shaders_prototype\\compute\\BVH\\screen_shader_accumulated.wgsl", device);
	
	raytracingKernelModule = ResourceManager::loadShaderModule("shaders_prototype\\compute\\BVH\\raytracing_bvh_accumulated.wgsl", device);

	initRaytraycingDataTexturesBindGroups();

	std::vector<WGPUBindGroupLayoutEntry> raytracingBindingLayout(6);


	createBindingLayout(0, sizeof(BVHNodeSSBO), raytracingBindingLayout[0], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Compute);
	//createBindingLayout(2, sizeof(LeafRopesUBO), raytracingBindingLayout[2], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Compute);
	createBindingLayout(1, sizeof(Sphere), raytracingBindingLayout[1], WGPUBufferBindingType_ReadOnlyStorage, WGPUShaderStage_Compute);
	createBindingLayout(2, sizeof(CameraUBO), raytracingBindingLayout[2], WGPUBufferBindingType_Uniform, WGPUShaderStage_Compute);
	createBindingLayout(3, sizeof(Config), raytracingBindingLayout[3], WGPUBufferBindingType_Uniform, WGPUShaderStage_Compute);
	
	WGPUTextureBindingLayout textureBindingLayoutColor{};
	textureBindingLayoutColor.sampleType = WGPUTextureSampleType_Float;
	createBindingLayout(4, 0, raytracingBindingLayout[4], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingBindingLayout[4].texture = textureBindingLayoutColor;
	raytracingBindingLayout[4].texture.multisampled = false;
	raytracingBindingLayout[4].texture.sampleType = WGPUTextureSampleType_Float;
	raytracingBindingLayout[4].texture.viewDimension = WGPUTextureViewDimension_Cube;

	WGPUSamplerBindingLayout cubeSamplerBindingLayout{};
	cubeSamplerBindingLayout.type = WGPUSamplerBindingType_Filtering;
	createBindingLayout(5, 0, raytracingBindingLayout[5], WGPUBufferBindingType_Undefined, WGPUShaderStage_Compute);
	raytracingBindingLayout[5].sampler = cubeSamplerBindingLayout;
	raytracingBindingLayout[5].sampler.nextInChain = nullptr;
	raytracingBindingLayout[5].sampler.type = WGPUSamplerBindingType_Filtering;

	WGPUBindGroupLayoutDescriptor raytracingBindGroupLayoutDesc = {};
	raytracingBindGroupLayoutDesc.entryCount = 6;
	raytracingBindGroupLayoutDesc.entries = raytracingBindingLayout.data();//&bindingLayout;
	WGPUBindGroupLayout raytracingBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &raytracingBindGroupLayoutDesc);

	initRaytracingBindGroup(raytracingBindGroup, raytracingBindGroupLayout);

	WGPUComputePipelineDescriptor raytracingPipelineDesc = {};
	raytracingPipelineDesc.nextInChain = nullptr;

	raytracingPipelineDesc.compute.module = raytracingKernelModule;
	raytracingPipelineDesc.compute.entryPoint = "main";

	
	std::vector<WGPUBindGroupLayout> raytracingBindGroupLayouts(2);
	raytracingBindGroupLayouts[0] = raytracingBindGroupLayout;
	raytracingBindGroupLayouts[1] = rayTracingDataTexturesBindLayout;
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	layoutDesc.bindGroupLayoutCount = 2;
	layoutDesc.bindGroupLayouts = raytracingBindGroupLayouts.data();

	

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
	createBindingLayout(1, sizeof(Config), screenBindingLayout[1], WGPUBufferBindingType_Uniform, WGPUShaderStage_Fragment);


	WGPUBindGroupLayoutDescriptor screenBindGroupLayoutDesc = {};
	screenBindGroupLayoutDesc.entryCount = 2;
	screenBindGroupLayoutDesc.entries = screenBindingLayout.data();//&bindingLayout;
	WGPUBindGroupLayout screenBindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &screenBindGroupLayoutDesc);

	initScreenBindGroup(screenBindGroup, screenBindGroupLayout);

	WGPURenderPipelineDescriptor screenPipelineDesc = {};
	screenPipelineDesc.nextInChain = nullptr;

	screenPipelineDesc.vertex.module = screenShaderModule;
	screenPipelineDesc.vertex.entryPoint = "vs_main";

	//fragment
	WGPUFragmentState fragmentState = {};
	fragmentState.nextInChain = nullptr;
	screenPipelineDesc.fragment = &fragmentState;
	fragmentState.module = screenShaderModule;
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

	initScreenDataTexturesBindGroups();
	initScreenAccumulationBindGroups();

	std::vector<WGPUBindGroupLayout> screenBindGroupLayouts(3);
	screenBindGroupLayouts[0] = screenBindGroupLayout;
	screenBindGroupLayouts[1] = screenDataTextureBindLayout;
	screenBindGroupLayouts[2] = screenAccumulationBindLayout;
	
	WGPUPipelineLayoutDescriptor screenLayoutDesc = {};
	//layoutDesc.nextInChain = nullptr;
	screenLayoutDesc.bindGroupLayoutCount = 3;
	screenLayoutDesc.bindGroupLayouts = screenBindGroupLayouts.data();

	WGPUPipelineLayout screenLayout = wgpuDeviceCreatePipelineLayout(device, &screenLayoutDesc);
	screenPipelineDesc.layout = screenLayout;

	screenPipeline = wgpuDeviceCreateRenderPipeline(device, &screenPipelineDesc);
	std::cout << "screenPipeline pipeline: " << screenPipeline << std::endl;
	std::cout << "BVH Nodes size: " << bvh->getBVHSSBOs().size() * sizeof(BVHNodeSSBO) << std::endl;
	std::cout << "Spheres size: " << bvh->getSpheres().size() * sizeof(Sphere) << std::endl;

}

void ComputeRendererBVHAccumulator::prepareCubemap() {
	
	WGPUTextureView *tv = nullptr;
	Cubemap cubemap = ResourceManager::loadCubemapTexture("resources\\Cubemaps\\Sky", device, tv);
	cubemapTexture = cubemap.texture;
	cubemapTextureView = cubemap.textureView;

	WGPUSamplerDescriptor samplerDescriptor{};
	samplerDescriptor.addressModeU = WGPUAddressMode_Repeat;
	samplerDescriptor.addressModeV = WGPUAddressMode_Repeat;
	samplerDescriptor.magFilter = WGPUFilterMode_Linear;
	samplerDescriptor.minFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.mipmapFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.maxAnisotropy = 1; 
	cubemapSampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);
	
}
