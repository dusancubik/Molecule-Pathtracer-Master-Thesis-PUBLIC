#pragma once
#include "../Pipelines/renderPipelineInterface.hpp"
#include "../KdTree/kdTreeInterface.hpp"

template<typename P, typename T>
class PipelineAbstractFactory {
	public:
		virtual std::shared_ptr<P> createRenderPipeline() = 0;
		virtual std::shared_ptr<T> createKdTree() = 0;
};