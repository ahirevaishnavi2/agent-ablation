from agents.preprocess import PreprocessAgent
from agents.chunking import ChunkingAgent
from agents.embedding import EmbeddingAgent
from agents.clustering import ClusteringAgent
from agents.summarizer import SummarizerAgent
from agents.critic import CriticAgent
from core.state import PipelineState

AGENT_REGISTRY = {
    "preprocess": PreprocessAgent,
    "chunking": ChunkingAgent,
    "embedding": EmbeddingAgent,
    "clustering": ClusteringAgent,
    "summarizer": SummarizerAgent,
    "critic": CriticAgent,
}

AGENT_ORDER = list(AGENT_REGISTRY.keys())


def build_agents(config: dict):
    agents_cfg = config.get("agents", {})
    agents = []
    for name in AGENT_ORDER:
        cfg = agents_cfg.get(name, {"enabled": False})
        agents.append(AGENT_REGISTRY[name](cfg))
    return agents


def run_pipeline(state: PipelineState) -> PipelineState:
    for agent in build_agents(state.config):
        state = agent.run(state)
    return state