/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: Skybox.hpp
 *
 *  Description:
 *  The Skybox class represents a skybox in the scene, storing its associated CubemapData (pixel values) and Texture (see Texture.hpp).
 * -----------------------------------------------------------------------------
 */
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