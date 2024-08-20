#pragma once

#include <iostream>
#include "application.h"
#include "glfw_factory.h"
#include "wgpu_context.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
#include <memory>
#include <vector>
class Timestamp {
public:
	Timestamp(WGPUDevice device);
	WGPUQuerySet getQuerySet() { return querySet; };
	WGPUBuffer getQueryBuffer() { return queryBuffer; };
	WGPUBuffer getStagingBuffer() { return stagingBuffer; };
	WGPURenderPassTimestampWrite getTimestampBegin() { return timestampBegin; };
	WGPURenderPassTimestampWrite getTimestampEnd() { return timestampEnd; };
	std::vector<WGPURenderPassTimestampWrite> getTimestamps() { return timestamps; };

	void* timestampsData = nullptr;
private:
	WGPUBuffer queryBuffer;
	WGPUBuffer stagingBuffer;


	WGPUQuerySet querySet;

	WGPUComputePassDescriptor computePassDescriptor;

	WGPURenderPassTimestampWrite timestampBegin;
	WGPURenderPassTimestampWrite timestampEnd;

	std::vector<WGPURenderPassTimestampWrite> timestamps;

	
};