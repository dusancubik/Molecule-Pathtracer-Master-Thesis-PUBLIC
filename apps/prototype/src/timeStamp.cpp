#include "timeStamp.hpp"

Timestamp<WGPURenderPassTimestampWrite>::Timestamp(WGPUDevice device) {
	

	int capacity = 2;

	WGPUQuerySetDescriptor querySetDescriptor{};
	querySetDescriptor.type = WGPUQueryType_Timestamp;
	querySetDescriptor.count = capacity;
	

	querySet = wgpuDeviceCreateQuerySet(device, &querySetDescriptor);
	
	WGPUBufferDescriptor bufferDesc{};
	bufferDesc.size = sizeof(int64_t) * capacity;
	bufferDesc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
	bufferDesc.mappedAtCreation = false;
	queryBuffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
	
	//stagin buffer
	WGPUBufferDescriptor stagingBufferDesc{};
	stagingBufferDesc.size = sizeof(int64_t) * capacity;
	stagingBufferDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
	stagingBufferDesc.mappedAtCreation = false;
	stagingBuffer = wgpuDeviceCreateBuffer(device, &stagingBufferDesc);

	//computePassDescriptor.qu
	timestampBegin.queryIndex = 0;
	timestampBegin.querySet = querySet;
	timestampBegin.location = WGPURenderPassTimestampLocation_Beginning;

	timestampEnd.queryIndex = 1;
	timestampEnd.querySet = querySet;
	timestampEnd.location = WGPURenderPassTimestampLocation_End;

	timestamps.push_back(timestampBegin);
	timestamps.push_back(timestampEnd);
	std::cout << "tissme stamp created!\n";
	//WGPUComputePassDescriptor computePassDescriptor1{ .timestampWrites = &renderPassTimestampWrite };
	//computePassDescriptor1.timestampWrites = &renderPassTimestampWrite;
}

Timestamp<WGPUComputePassTimestampWrite>::Timestamp(WGPUDevice device) {


	int capacity = 2;

	WGPUQuerySetDescriptor querySetDescriptor{};
	querySetDescriptor.type = WGPUQueryType_Timestamp;
	querySetDescriptor.count = capacity;


	querySet = wgpuDeviceCreateQuerySet(device, &querySetDescriptor);

	WGPUBufferDescriptor bufferDesc{};
	bufferDesc.size = sizeof(int64_t) * capacity;
	bufferDesc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
	bufferDesc.mappedAtCreation = false;
	queryBuffer = wgpuDeviceCreateBuffer(device, &bufferDesc);

	//stagin buffer
	WGPUBufferDescriptor stagingBufferDesc{};
	stagingBufferDesc.size = sizeof(int64_t) * capacity;
	stagingBufferDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
		stagingBufferDesc.mappedAtCreation = false;
	stagingBuffer = wgpuDeviceCreateBuffer(device, &stagingBufferDesc);

	//computePassDescriptor.qu
	timestampBegin.queryIndex = 0;
	timestampBegin.querySet = querySet;
	timestampBegin.location = WGPUComputePassTimestampLocation_Beginning;

	timestampEnd.queryIndex = 1;
	timestampEnd.querySet = querySet;
	timestampEnd.location = WGPUComputePassTimestampLocation_End;

	timestamps.push_back(timestampBegin);
	timestamps.push_back(timestampEnd);
	std::cout << "tissme stamp created!\n";
	//WGPUComputePassDescriptor computePassDescriptor1{ .timestampWrites = &renderPassTimestampWrite };
	//computePassDescriptor1.timestampWrites = &renderPassTimestampWrite;
}