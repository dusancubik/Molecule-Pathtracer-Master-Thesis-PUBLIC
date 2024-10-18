#include "../../include/Renderer/Texture.hpp"
#pragma once



#include "../../analyst/src/stb_image.h"

Texture::Texture(WGPUDevice _device, int width, int height, WGPUTextureFormat format) {

	device = _device;

	WGPUTextureDescriptor textureDesc{};
	textureDesc.nextInChain = nullptr;

	textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
	textureDesc.format = format;
	textureDesc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_TextureBinding;
	textureDesc.sampleCount = 1;
	textureDesc.mipLevelCount = 1;
	texture = wgpuDeviceCreateTexture(device, &textureDesc);

	WGPUTextureViewDescriptor textureViewDesc{};
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 1;
	textureView = wgpuTextureCreateView(texture, &textureViewDesc);


	WGPUSamplerDescriptor samplerDescriptor{};
	samplerDescriptor.addressModeU = WGPUAddressMode_Repeat;
	samplerDescriptor.addressModeV = WGPUAddressMode_Repeat;
	samplerDescriptor.magFilter = WGPUFilterMode_Linear;
	samplerDescriptor.minFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.mipmapFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.maxAnisotropy = 1;

	sampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);
	
}

Texture::Texture(WGPUDevice _device, int width, int height, WGPUTextureFormat format, CubemapData cubemapData) {
	device = _device;
	WGPUTextureDescriptor textureDesc{};
	textureDesc.nextInChain = nullptr;
	//textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.dimension = WGPUTextureDimension_2D;
	textureDesc.size = { 1280, 720, 6 };
	textureDesc.format = format;
	textureDesc.usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
	textureDesc.sampleCount = 1;
	textureDesc.mipLevelCount = 1;
	textureDesc.size = cubemapData.size;
	texture = wgpuDeviceCreateTexture(device, &textureDesc);
	// [...]
	WGPUExtent3D cubemapLayerSize = { cubemapData.size.width , cubemapData.size.height , 1 };
	for (uint32_t layer = 0; layer < 6; ++layer) {
		WGPUOrigin3D origin = { 0, 0, layer };

		writeMipMaps(device, texture, cubemapLayerSize, textureDesc.mipLevelCount, cubemapData.pixelData[layer], origin);

		
		stbi_image_free(cubemapData.pixelData[layer]);
	}

	WGPUTextureViewDescriptor textureViewDesc{};
	textureViewDesc.dimension = WGPUTextureViewDimension_Cube;
	textureViewDesc.mipLevelCount = 1;
	textureViewDesc.arrayLayerCount = 6;
	textureView = wgpuTextureCreateView(texture, &textureViewDesc);


	WGPUSamplerDescriptor samplerDescriptor{};
	samplerDescriptor.addressModeU = WGPUAddressMode_Repeat;
	samplerDescriptor.addressModeV = WGPUAddressMode_Repeat;
	samplerDescriptor.magFilter = WGPUFilterMode_Linear;
	samplerDescriptor.minFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.mipmapFilter = WGPUFilterMode_Nearest;
	samplerDescriptor.maxAnisotropy = 1;
	sampler = wgpuDeviceCreateSampler(device, &samplerDescriptor);
}

void Texture::writeMipMaps(
	WGPUDevice device,
	WGPUTexture texture,
	WGPUExtent3D textureSize,
	[[maybe_unused]] uint32_t mipLevelCount, // not used yet
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