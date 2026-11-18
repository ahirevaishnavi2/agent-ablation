from agents.base import BaseAgent
from core.state import PipelineState


class CriticAgent(BaseAgent):
    name = "critic"

    def _run(self, state: PipelineState) -> PipelineState:
        if not state.summaries:
            return state

        lengths = [len(s["summary"].split()) for s in state.summaries]
        input_lengths = [s["size"] for s in state.summaries]

        state.metrics["summary_avg_words"] = round(sum(lengths) / len(lengths), 2)
        state.metrics["summary_min_words"] = min(lengths)
        state.metrics["summary_max_words"] = max(lengths)
        state.metrics["summary_compression"] = round(
            sum(lengths) / max(1, sum(input_lengths) * 50), 4
        )
        state.metrics["empty_summaries"] = sum(1 for s in state.summaries if not s["summary"].strip())
        return state