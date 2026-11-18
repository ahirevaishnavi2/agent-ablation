import re
from collections import Counter

import numpy as np
from agents.base import BaseAgent
from core.state import PipelineState

_STOP = set(
    "a an the and or but if then else of to in on at for with by from as is are was were "
    "be been being it its this that these those i you he she they we my your our their "
    "not no so do does did have has had will would can could should about into over".split()
)


def split_sentences(text: str) -> list[str]:
    # Remove quote markers (>, |>, etc.) from the start of lines
    text = re.sub(r"(?m)^\s*[>|]+\s*", "", text)
    # Split by punctuation
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sents


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    sents = split_sentences(text)
    
    # Deduplicate sentences while preserving order
    seen = set()
    unique_sents = []
    for s in sents:
        if s not in seen:
            unique_sents.append(s)
            seen.add(s)
            
    sents = unique_sents

    if len(sents) <= max_sentences:
        return " ".join(sents)

    words = re.findall(r"[a-z']+", text.lower())
    freq = Counter(w for w in words if w not in _STOP)
    if not freq:
        return " ".join(sents[:max_sentences])

    max_f = max(freq.values())
    scores = []
    for s in sents:
        ws = re.findall(r"[a-z']+", s.lower())
        if not ws:
            scores.append(0.0)
            continue
        scores.append(sum(freq[w] / max_f for w in ws if w in freq) / len(ws))

    top_indices = np.argsort(scores)[::-1][:max_sentences]
    # Sort back to original order so the summary reads naturally
    return " ".join(sents[i] for i in sorted(top_indices))


class SummarizerAgent(BaseAgent):
    name = "summarizer"

    def _run(self, state: PipelineState) -> PipelineState:
        max_sents = int(self.cfg.get("max_sentences", 3))
        max_words = int(self.cfg.get("max_input_words", 1500))

        by_cluster = {}
        for chunk, label in zip(state.chunks, state.cluster_labels):
            by_cluster.setdefault(int(label), []).append(chunk["text"])

        summaries = []
        for cluster_id, texts in sorted(by_cluster.items()):
            joined = " ".join(texts).split()[:max_words]
            summary = extractive_summary(" ".join(joined), max_sents)
            summaries.append(
                {"cluster": cluster_id, "size": len(texts), "summary": summary}
            )

        state.summaries = summaries
        return state