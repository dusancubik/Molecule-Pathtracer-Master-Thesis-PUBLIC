/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: ResourceManager.hpp
 *
 *  Description:
 *  The ResourceManager class is responsible for loading resources (shaders, molecules, cubemap) from a file.
 *  Method loadCubemapData and writeMipMaps is taken from Elie Michel's LearnWebGPU-Code repository
 *  (https://github.com/eliemichel/LearnWebGPU-Code/blob/step117/ResourceManager.cpp) which is licensed under the MIT License.
 *  Method loadShaderModule is taken from Elie Michel's LearnWebGPU-Code repository 
 *  (https://github.com/eliemichel/LearnWebGPU-Code/blob/step037-vanilla/ResourceManager.cpp) which is licensed under the MIT License.
 * 
 * -----------------------------------------------------------------------------
 */
#pragma once

#include "wgpu_context.h"
#include "glm/glm.hpp"
#include "atom/atom.hpp"
#include <vector>
#include <filesystem>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <regex>

//#define STB_IMAGE_IMPLEMENTATION
#include "../../pbr_renderer/include/Utils/stb_image.h" 

struct Cubemap {
	WGPUTexture texture;
	WGPUTextureView textureView;
};

class ResourceManager {
public:
	
	using path = std::filesystem::path;
	using vec3 = glm::vec3;
	using vec2 = glm::vec2;


	struct VertexAttributes {
		vec3 position;
		vec3 normal;
		vec3 color;
		vec2 uv;
	};


	static WGPUShaderModule loadShaderModule(const path& path, WGPUDevice device);


	static std::vector<SphereCPU*> loadAtoms(const path& path);

	static void parseLineToAtom(std::string& line, SphereCPU* atom);

	static void loadPixelData(const path& path, unsigned char* pixelData);
	static Cubemap loadCubemapTexture(const path& path, WGPUDevice device, WGPUTextureView* pTextureView);
	static void writeMipMaps(
		WGPUDevice device,
		WGPUTexture texture,
		WGPUExtent3D textureSize,
		[[maybe_unused]] uint32_t mipLevelCount,
		const unsigned char* pixelData
		, WGPUOrigin3D origin = { 0, 0, 0 });
};