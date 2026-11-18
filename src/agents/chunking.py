from agents.base import BaseAgent
from core.state import PipelineState


def chunk_words(text: str, size: int, overlap: float) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, int(size * (1.0 - overlap)))
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


class ChunkingAgent(BaseAgent):
    name = "chunking"

    def _run(self, state: PipelineState) -> PipelineState:
        size = int(self.cfg.get("size", 256))
        overlap = float(self.cfg.get("overlap", 0.2))

        chunks = []
        for doc_id, text in enumerate(state.texts):
            for i, piece in enumerate(chunk_words(text, size, overlap)):
                chunks.append({"doc_id": doc_id, "chunk_id": i, "text": piece})

        state.chunks = chunks
        state.metrics["n_chunks"] = len(chunks)
        return state