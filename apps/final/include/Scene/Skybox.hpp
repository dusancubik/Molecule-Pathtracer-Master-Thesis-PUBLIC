#pragma once
#include "../Renderer/Texture.hpp"

class Skybox {
	public:
		Skybox(CubemapData _cubemapData);
		Texture* getTexture() { return texture; }
		CubemapData getCubemapData() { return cubemapData; }
	private:
		CubemapData cubemapData;
		Texture* texture;
};