#pragma once
#include "renderPipelineInterface.hpp"
#include "../KdTree/kdTree.hpp"

class KdTreeRenderPipeline : public RenderPipelineInterface {
	public:
		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) override;

		void render(WGPUTextureView &nextTexture) override;

        int getFrameTimeNS() { return frameTimeNS; }

        std::shared_ptr<Camera> getCamera() { return camera; };

        void setKdTree(std::shared_ptr<KdTree> _kdTree) { kdTree = _kdTree; }
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
        //kdTree
        std::shared_ptr<KdTree> kdTree;

        WGPUBuffer kdTreeStorageBuffer = nullptr;
        WGPUBuffer leavesStorageBuffer = nullptr;


        void createBindingLayout(uint32_t binding, uint64_t minBindingSize, WGPUBindGroupLayoutEntry& bindingLayout, WGPUBufferBindingType bufferType, WGPUShaderStageFlags shaderFlags);
};