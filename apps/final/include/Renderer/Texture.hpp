#pragma once
#include "wgpu_context.h"
#include <array>
#include "../Utils/stb_image.h"


struct CubemapData {
	WGPUExtent3D size;
	std::array<uint8_t*, 6> pixelData;
};

class Texture {
	public:
		Texture(WGPUDevice _device, int width, int height, WGPUTextureFormat format);

		Texture(WGPUDevice _device, int width, int height, WGPUTextureFormat format, CubemapData cubemapData);
		//bool init(int width, int height, WGPUTextureFormat format);

		WGPUTextureView getTextureView() { return textureView; }
		WGPUSampler getSampler() { return sampler; }

		WGPUTexture texture;
		WGPUTextureView textureView;
		WGPUSampler sampler;

	private:
		void writeMipMaps(WGPUDevice device, WGPUTexture texture, WGPUExtent3D textureSize,	[[maybe_unused]] uint32_t mipLevelCount,  const unsigned char* pixelData,
			WGPUOrigin3D origin);
		WGPUDevice device;
};