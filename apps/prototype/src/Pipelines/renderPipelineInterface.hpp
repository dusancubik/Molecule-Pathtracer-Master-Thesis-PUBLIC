#pragma once
#include <iostream>
#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
//#include "ResourceManager.hpp"
#include "camera.hpp"
#include "timeStamp.hpp"
//#include "renderPipeline.hpp"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE //this needs to be set because of vulkan??
#define GLM_FORCE_LEFT_HANDED
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "../atom/sphere.hpp"
#include <random>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <functional>

#include "../ResourceManager.cpp"
#include "../camera.cpp"
#include "../timeStamp.cpp"
//#include "../kdTree.cpp"



class RenderPipelineInterface {
	public:
		virtual ~RenderPipelineInterface() {}

		virtual void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) = 0;

		virtual void render(WGPUTextureView& nextTexture) = 0;

		virtual std::shared_ptr<Camera> getCamera() = 0;

};