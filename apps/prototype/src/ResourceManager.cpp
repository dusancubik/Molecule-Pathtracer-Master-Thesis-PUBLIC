#include "ResourceManager.hpp"



WGPUShaderModule ResourceManager::loadShaderModule(const path& path, WGPUDevice device) {
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cout << "Cannot load module" << std::endl;
		return nullptr;
	}
	file.seekg(0, std::ios::end);
	size_t size = file.tellg();
	std::string shaderSource(size, ' ');
	file.seekg(0);
	file.read(shaderSource.data(), size);

	WGPUShaderModuleWGSLDescriptor shaderCodeDesc = {};
	shaderCodeDesc.chain.next = nullptr;
	shaderCodeDesc.chain.sType = WGPUSType::WGPUSType_ShaderModuleWGSLDescriptor;
	shaderCodeDesc.source = shaderSource.c_str();
	WGPUShaderModuleDescriptor shaderDesc = {};
	shaderDesc.nextInChain = &shaderCodeDesc.chain;
#ifdef WEBGPU_BACKEND_WGPU
	shaderDesc.hintCount = 0;
	shaderDesc.hints = nullptr;
#endif

	return wgpuDeviceCreateShaderModule(device, &shaderDesc);
}

std::vector<SphereCPU*> ResourceManager::loadAtoms(const path& path) {
	std::ifstream file;
	file.open(path);

	std::vector<SphereCPU*> atoms;

	if (!file.is_open()) {
		std::cerr << "Error opening file" << std::endl;
		return atoms;
	}


	std::string line;
	glm::vec3 minAABB = glm::vec3(9999999.f);
	glm::vec3 maxAABB = glm::vec3(-9999999.f);
	while (getline(file, line) ) {
		//std::stringstream ss(line);
		//std::string word;

		std::string firstWord = line.substr(0, line.find_first_of(" "));
		if (firstWord == "ATOM") {
			std::cout << "ATOM\n";
			SphereCPU* atom = new SphereCPU;
			parseLineToAtom(line, atom);
			//break;
			if (atom->radius == 1.f) { 
				minAABB = glm::min(minAABB,atom->origin);
				maxAABB = glm::max(maxAABB, atom->origin);
				atoms.push_back(atom); 
			}
		}
		else {
			std::cout << "NOT ATOM\n";
		}
	}
	glm::vec3 midAABB = (minAABB + maxAABB) / 2.f;
	std::cerr << "minAABB: ("<< minAABB.x<<"," << minAABB.y << "," << minAABB.z << ")" << std::endl;
	std::cerr << "maxAABB: (" << maxAABB.x << "," << maxAABB.y << "," << maxAABB.z << ")" << std::endl;
	std::cerr << "-----------NEW-----------" << std::endl;
	minAABB = minAABB - midAABB;
	maxAABB = maxAABB - midAABB;
	std::cerr << "midAABB: (" << midAABB.x << "," << midAABB.y << "," << midAABB.z << ")" << std::endl;
	std::cerr << "minAABB: (" << minAABB.x << "," << minAABB.y << "," << minAABB.z << ")" << std::endl;
	std::cerr << "maxAABB: (" << maxAABB.x << "," << maxAABB.y << "," << maxAABB.z << ")" << std::endl;
	
	for (int i = 0;i < atoms.size();i++) atoms[i]->origin -= midAABB;
	return atoms;
}

void ResourceManager::loadPixelData(const path& path, unsigned char* pixelData) {
	int width, height, channels;
	pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4 /* force 4 channels */);
}

void ResourceManager::parseLineToAtom(std::string& line, SphereCPU* atom) {

	std::regex pdb_regex(R"(^ATOM\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S)\s+(\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+(\S+))");
	std::smatch matches;
	float scale = 1.f;
	if (std::regex_search(line, matches, pdb_regex)) {
		atom->color = glm::vec4(.3f, 0.f, 0.f, 1.f);
		atom->radius = 1.f;
		atom->origin = glm::vec3(scale * std::stof(matches[6].str()), scale * std::stof(matches[7].str()), scale * std::stof(matches[8].str()));
	}


	
}

Cubemap ResourceManager::loadCubemapTexture(const path& path, WGPUDevice device, WGPUTextureView* pTextureView) {
	const char* cubemapPaths[] = {
		"px.png",
		"nx.png",
		"py.png",
		"ny.png",
		"pz.png",
		"nz.png",
	};

	
	WGPUExtent3D cubemapSize = { 0, 0, 6 };
	std::array<uint8_t*, 6> pixelData;
	for (uint32_t layer = 0; layer < 6; ++layer) {
		int width, height, channels;
		auto p = path / cubemapPaths[layer];
		pixelData[layer] = stbi_load(p.string().c_str(), &width, &height, &channels, 4);
		if (nullptr == pixelData[layer]) throw std::runtime_error("Could not load input texture!");
		if (layer == 0) {
			cubemapSize.width = (uint32_t)width;
			cubemapSize.height = (uint32_t)height;
		}
		else {
			if (cubemapSize.width != (uint32_t)width || cubemapSize.height != (uint32_t)height)
				throw std::runtime_error("All cubemap faces must have the same size!");
		}
	}

	
	WGPUTextureDescriptor textureDesc{};
	textureDesc.nextInChain = nullptr;
	//textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.size = { 1280, 720, 6 };
	textureDesc.format = WGPUTextureFormat_RGBA8Unorm;
	textureDesc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
	textureDesc.sampleCount = 1;
	textureDesc.mipLevelCount = 1;
	textureDesc.size = cubemapSize;
	WGPUTexture texture = wgpuDeviceCreateTexture(device, &textureDesc);
	// [...]
	WGPUExtent3D cubemapLayerSize = { cubemapSize.width , cubemapSize.height , 1 };
	for (uint32_t layer = 0; layer < 6; ++layer) {
		WGPUOrigin3D origin = { 0, 0, layer };

		writeMipMaps(device, texture, cubemapLayerSize, textureDesc.mipLevelCount, pixelData[layer], origin);

		
		stbi_image_free(pixelData[layer]);
	}

	WGPUTextureViewDescriptor textureViewDesc{};
	textureViewDesc.dimension = WGPUTextureViewDimension_Cube;
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 6;
	WGPUTextureView cubemapTexture_view = wgpuTextureCreateView(texture, &textureViewDesc);
    
	return Cubemap(texture, cubemapTexture_view);
}

void ResourceManager::writeMipMaps(
	WGPUDevice device,
	WGPUTexture texture,
	WGPUExtent3D textureSize,
	[[maybe_unused]] uint32_t mipLevelCount, 
	const unsigned char* pixelData,
	WGPUOrigin3D origin)
{
	WGPUImageCopyTexture destination = {};
	destination.texture = texture;
	destination.mipLevel = 0;
	destination.origin = origin;
	destination.aspect = WGPUTextureAspect_All;

	WGPUTextureDataLayout source = {};
	source.offset = 0;
	source.bytesPerRow = 4 * textureSize.width;
	source.rowsPerImage = textureSize.height;

	WGPUQueue queue = wgpuDeviceGetQueue(device);
	wgpuQueueWriteTexture(queue, &destination, pixelData, 4 * textureSize.width * textureSize.height, &source, &textureSize);
	wgpuQueueRelease(queue);
}