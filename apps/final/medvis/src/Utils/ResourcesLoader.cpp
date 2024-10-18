#include "../../include/Utils/ResourcesLoader.hpp"
#pragma once




void ResourcesLoader::loadPixelData(const path& path, unsigned char* pixelData) {
	int width, height, channels;
	pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4 /* force 4 channels */);
}

CubemapData ResourcesLoader::loadCubemapData(const path& path) {
	const char* cubemapPaths[] = {
		"cubemap-posX.png",
		"cubemap-negX.png",
		"cubemap-posY.png",
		"cubemap-negY.png",
		"cubemap-posZ.png",
		"cubemap-negZ.png",
	};

	
	WGPUExtent3D cubemapSize = { 0, 0, 6 };
	std::array<uint8_t*, 6> pixelData;
	for (uint32_t layer = 0; layer < 6; ++layer) {
		int width, height, channels;
		auto p = path / cubemapPaths[layer];
		pixelData[layer] = stbi_load(p.string().c_str(), &width, &height, &channels, 4 /* force 4 channels */);
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

	return CubemapData(cubemapSize,pixelData);
}

void ResourcesLoader::test() {

}