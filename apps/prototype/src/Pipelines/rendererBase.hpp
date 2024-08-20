#pragma once
#include "renderPipelineInterface.hpp"


class RendererBase : public RenderPipelineInterface {
	public:
		~RendererBase() {}

		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format);

		void render(WGPUTextureView& nextTexture) {}

		int getFrameTimeNS() { return frameTimeNS; }

		std::shared_ptr<Camera> getCamera() { return nullptr; }

		void createBindingLayout(uint32_t binding, uint64_t minBindingSize, WGPUBindGroupLayoutEntry& bindingLayout, WGPUBufferBindingType bufferType, WGPUShaderStageFlags shaderFlags);
		void createBindGroupEntry(WGPUBindGroupEntry& bindGroupEntry, uint32_t binding, WGPUBuffer buffer, uint64_t offset, uint64_t size);
		void createBuffer(WGPUBuffer& buffer, WGPUBufferUsageFlags flags, uint64_t size, const void* data);
		void initDepthBuffer();
	protected:
		//Timestamp
		std::shared_ptr<Timestamp> timestamp;
		static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
		int frameTimeNS = 0;

		WGPUDevice device;
		WGPUQueue queue;
		WGPUTextureFormat swap_chain_default_format;

		//
		std::vector<SphereCPU*> spheres;

		// Depth Buffer
		WGPUTextureFormat depthTextureFormat = WGPUTextureFormat_Depth24Plus;
		WGPUTexture depthTexture = nullptr;
		WGPUTextureView depthTextureView = nullptr;
		WGPUDepthStencilState depthStencilState{};

		virtual void initUniforms(){}
		virtual void initBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) {}
};