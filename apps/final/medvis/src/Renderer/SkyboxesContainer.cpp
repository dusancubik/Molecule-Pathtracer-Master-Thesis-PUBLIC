#include "../../include/Renderer/SkyboxesContainer.hpp"


SkyboxesContainer::SkyboxesContainer(WGPUDevice _device) {
	device = _device;

	layoutEntries = {
		BindGroupLayoutEntry(0, WGPUShaderStage_Compute, WGPUTextureViewDimension_Cube, WGPUTextureSampleType_Float),
		BindGroupLayoutEntry(1, WGPUShaderStage_Compute, WGPUSamplerBindingType_Filtering) //sampler
	};
}

void SkyboxesContainer::addSkybox(CubemapData cubemapData, std::string skyboxName) {
	Texture* skyboxTexture = new Texture(device, 0, 0, WGPUTextureFormat_RGBA8Unorm, cubemapData);

	BindGroup* bindGroup = new BindGroup(device, layoutEntries);

	bindGroup->addEntry(new TextureBindGroupEntry(0, skyboxTexture->getTextureView()));
	bindGroup->addEntry(new SamplerBindGroupEntry(1, skyboxTexture->getSampler()));

	bindGroup->finalize();

	skyboxNames.push_back(skyboxName);
	skyboxesBindGroupsMap[skyboxName] = bindGroup;
}

BindGroup* SkyboxesContainer::getSkyboxBindGroup(std::string skyboxName) {
	return skyboxesBindGroupsMap[skyboxName];
}