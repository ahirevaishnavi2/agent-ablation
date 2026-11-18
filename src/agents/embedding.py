import numpy as np
from agents.base import BaseAgent
from core.state import PipelineState


class EmbeddingAgent(BaseAgent):
    name = "embedding"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg.get("model", "all-MiniLM-L6-v2"))
        return self._model

    def _run(self, state: PipelineState) -> PipelineState:
        model = self._load()
        texts = [c["text"] for c in state.chunks]
        emb = model.encode(
            texts,
            batch_size=int(self.cfg.get("batch_size", 64)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        state.embeddings = np.asarray(emb, dtype=np.float32)
        state.metrics["embedding_dim"] = int(state.embeddings.shape[1])
        return state