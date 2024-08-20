#pragma once
#include "pipelineAbstractFactory.hpp"
#include "../KdTree/kdTree.hpp"
#include "../Pipelines/kdTreeRenderPipeline.hpp"


class KdTreeStandardFactory : public PipelineAbstractFactory<KdTreeRenderPipeline, KdTree>{
	public:
		std::shared_ptr<KdTreeRenderPipeline> createRenderPipeline();
		std::shared_ptr<KdTree> createKdTree();
};