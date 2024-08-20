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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" 

struct Cubemap {
	WGPUTexture texture;
	WGPUTextureView textureView;
};

class ResourceManager {
public:
	// (Just aliases to make notations lighter)
	using path = std::filesystem::path;
	using vec3 = glm::vec3;
	using vec2 = glm::vec2;

	/**
	 * A structure that describes the data layout in the vertex buffer,
	 * used by loadGeometryFromObj and used it in `sizeof` and `offsetof`
	 * when uploading data to the GPU.
	 */
	struct VertexAttributes {
		vec3 position;
		vec3 normal;
		vec3 color;
		vec2 uv;
	};

	// Load a shader from a WGSL file into a new shader module
	static WGPUShaderModule loadShaderModule(const path& path, WGPUDevice device);

	// Load an 3D mesh from a standard .obj file into a vertex data buffer
	//static bool loadGeometryFromObj(const path& path, std::vector<VertexAttributes>& vertexData);
	//geometry from txt
	static bool loadGeometry(const path& path, std::vector<float>& pointData, std::vector<uint16_t>& indexData, int dimensions);
	
	static std::vector<SphereCPU*> loadAtoms(const path& path);

	static void parseLineToAtom(std::string& line, SphereCPU* atom);

	static void loadPixelData(const path& path, unsigned char* pixelData);
	static Cubemap loadCubemapTexture(const path& path, WGPUDevice device, WGPUTextureView* pTextureView);
	static void writeMipMaps(
		WGPUDevice device,
		WGPUTexture texture,
		WGPUExtent3D textureSize,
		[[maybe_unused]] uint32_t mipLevelCount, // not used yet
		const unsigned char* pixelData
		, WGPUOrigin3D origin = { 0, 0, 0 });
};