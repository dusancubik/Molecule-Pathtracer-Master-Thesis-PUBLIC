#include "pipelineAbstractFactory.hpp"

std::shared_ptr<KdTreeRenderPipeline> KdTreeStandardFactory::createRenderPipeline() {
	return std::make_shared<KdTreeRenderPipeline>();
}
std::shared_ptr<KdTree> KdTreeStandardFactory::createKdTree() {
	return std::make_shared<KdTree>();
}