#pragma once
#include "rendererBase.hpp"
#include "../KdTree/kdTreeRopes.hpp"

class KdTreeRopesRenderPipeline : public RendererBase {
	public:
		void init(std::vector<SphereCPU*> _spheres, WGPUDevice _device, WGPUQueue _queue, WGPUTextureFormat _swap_chain_default_format) override;

		void render(WGPUTextureView &nextTexture) override;

        std::shared_ptr<Camera> getCamera() override { return camera; };

        void setKdTree(std::shared_ptr<KdTreeRopes> _kdTree)  { kdTree = _kdTree; }
	private:
        // Render Pipeline
        //WGPUBindGroupLayout bindGroupLayout = nullptr;
        WGPUShaderModule shaderModule = nullptr;
        WGPURenderPipeline pipeline = nullptr;

        //BindGroup
        WGPUBindGroup bindGroup = nullptr;

        // Uniforms
        WGPUBuffer uniformBuffer = nullptr;

        //Spheres
        //std::vector<SphereCPU*> spheres;
        WGPUBuffer spheresStorageBuffer = nullptr;
        //Timestamp
        //std::shared_ptr<Timestamp> timestamp;

        //Camera
        std::shared_ptr<Camera> camera;
        bool initCamera();

        //static void readBufferMap(WGPUBufferMapAsyncStatus status, void* userdata);
        //int frameTimeNS = 0;
        //kdTree
        std::shared_ptr<KdTreeRopes> kdTree;

        WGPUBuffer kdTreeStorageBuffer = nullptr;
        WGPUBuffer leavesStorageBuffer = nullptr;


       // void createBindingLayout(uint32_t binding, uint64_t minBindingSize, WGPUBindGroupLayoutEntry& bindingLayout, WGPUBufferBindingType bufferType, WGPUShaderStageFlags shaderFlags);
    protected:
            void initUniforms() override;
            void initBindGroup(WGPUBindGroup& bindGroup, WGPUBindGroupLayout bindGroupLayout) override;
};