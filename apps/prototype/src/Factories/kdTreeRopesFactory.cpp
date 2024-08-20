#include "kdTreeRopesFactory.hpp"

std::shared_ptr<KdTreeRopesRenderPipeline> KdTreeRopesFactory::createRenderPipeline() {
	return std::make_shared<KdTreeRopesRenderPipeline>();
}
std::shared_ptr<KdTreeRopes> KdTreeRopesFactory::createKdTree() {
	return std::make_shared<KdTreeRopes>();
}