#pragma once
#include "pipelineAbstractFactory.hpp"
#include "../KdTree/kdTreeRopes.hpp"
#include "../Pipelines/kdTreeRopesRenderPipeline.hpp"


class KdTreeRopesFactory : public PipelineAbstractFactory<KdTreeRopesRenderPipeline, KdTreeRopes>{
	public:
		std::shared_ptr<KdTreeRopesRenderPipeline> createRenderPipeline();
		std::shared_ptr<KdTreeRopes> createKdTree();
};