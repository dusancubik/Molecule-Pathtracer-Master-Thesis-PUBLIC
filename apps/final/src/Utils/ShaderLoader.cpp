#include "../../include/Utils/ShaderLoader.hpp"

WGPUShaderModule ShaderLoader::loadShaderModule(const path& path, WGPUDevice device) {
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