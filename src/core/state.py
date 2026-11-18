from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class PipelineState:
    config: dict
    df: Optional[pd.DataFrame] = None
    texts: Optional[list[str]] = None
    labels: Optional[list[str]] = None
    chunks: Optional[list[dict]] = None          # [{"doc_id", "chunk_id", "text"}]
    embeddings: Any = None                       # np.ndarray (n_chunks, dim)
    cluster_labels: Any = None                   # np.ndarray (n_chunks,)
    summaries: Optional[list[dict]] = None       # [{"cluster", "size", "summary"}]
    metrics: dict = field(default_factory=dict)