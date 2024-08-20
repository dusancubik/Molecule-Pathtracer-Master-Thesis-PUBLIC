#pragma once
#include "wgpu_context.h"
#include <filesystem>
#include <fstream>

class ShaderLoader {
public:
	using path = std::filesystem::path;
	static WGPUShaderModule loadShaderModule(const path& path, WGPUDevice device);
};