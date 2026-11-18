import time
from abc import ABC, abstractmethod

from core.state import PipelineState


class BaseAgent(ABC):
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.enabled = self.cfg.get("enabled", True)

    @abstractmethod
    def _run(self, state: PipelineState) -> PipelineState:
        ...

    def run(self, state: PipelineState) -> PipelineState:
        if not self.enabled:
            return state
        t0 = time.perf_counter()
        state = self._run(state)
        state.metrics[f"time_{self.name}"] = round(time.perf_counter() - t0, 2)
        return state