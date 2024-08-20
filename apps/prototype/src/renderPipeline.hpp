#pragma once
#include "Pipelines/renderPipelineInterface.hpp"


class RenderPipeline : public RenderPipelineInterface {
	public:
		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) override;

		void render(WGPUTextureView &nextTexture) override;

        int getFrameTimeNS() { return frameTimeNS; }

        std::shared_ptr<Camera> getCamera() { return camera; };
	private:
        //from app
        WGPUDevice device;
        WGPUQueue queue;
        WGPUTextureFormat swap_chain_default_format;
        
        //webgpu::WGPUContext context;

        void initDepthBuffer();
        void initUniforms();
        void initBindGroup();
        // Depth Buffer
        WGPUTextureFormat depthTextureFormat = WGPUTextureFormat_Depth24Plus;
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthTextureView = nullptr;
        WGPUDepthStencilState depthStencilState{};
        // Render Pipeline
        WGPUBindGroupLayout bindGroupLayout = nullptr;
        WGPUShaderModule shaderModule = nullptr;
        WGPURenderPipeline pipeline = nullptr;

        //BindGroup
        WGPUBindGroup bindGroup = nullptr;

        // Uniforms
        WGPUBuffer uniformBuffer = nullptr;

        //Spheres
        std::vector<SphereCPU*> spheres;
        WGPUBuffer spheresStorageBuffer = nullptr;
        //Timestamp
        std::shared_ptr<Timestamp> timestamp;

        //Camera
        std::shared_ptr<Camera> camera;
        bool initCamera();

        static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
        int frameTimeNS = 0;
};