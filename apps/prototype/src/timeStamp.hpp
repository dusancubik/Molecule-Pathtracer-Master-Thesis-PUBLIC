/*
 * -----------------------------------------------------------------------------
 *  Author: Dusan Cubik
 *  Project: Physically Based Renderer for WebGPU (Prototype)
 *  Institution: Masaryk University
 *  Date: 16. 12. 2024
 *  File: timeStamp.hpp
 *
 *  Description:
 *  This Timestamp class is made to generate timestamps, helping to measure how long a shader takes to run.
 * -----------------------------------------------------------------------------
 */
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

template <typename T>
class Timestamp {
public:
	Timestamp(WGPUDevice device);
	WGPUQuerySet getQuerySet() { return querySet; };
	WGPUBuffer getQueryBuffer() { return queryBuffer; };
	WGPUBuffer getStagingBuffer() { return stagingBuffer; };
	T getTimestampBegin() { return timestampBegin; };
	T getTimestampEnd() { return timestampEnd; };
	std::vector<T> getTimestamps() { return timestamps; };

	void* timestampsData = nullptr;
private:
	WGPUBuffer queryBuffer;
	WGPUBuffer stagingBuffer;


	WGPUQuerySet querySet;

	WGPUComputePassDescriptor computePassDescriptor;

	T timestampBegin;
	T timestampEnd;

	std::vector<T> timestamps;

	
};