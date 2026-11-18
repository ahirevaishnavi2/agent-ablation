import numpy as np
import pandas as pd
from agents.base import BaseAgent
from core.state import PipelineState


def purity(y_true, y_pred) -> float:
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    return float(df.groupby("p")["t"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))


class ClusteringAgent(BaseAgent):
    name = "clustering"

    def _run(self, state: PipelineState) -> PipelineState:
        algo = self.cfg.get("algo", "kmeans").lower()
        X = state.embeddings
        seed = int(state.config.get("seed", 42))

        if algo == "kmeans":
            from sklearn.cluster import KMeans
            k = int(self.cfg.get("k", 15))
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = model.fit_predict(X)
        elif algo == "hdbscan":
            import hdbscan
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(self.cfg.get("min_cluster_size", 50)),
                metric="euclidean",
            )
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unknown clustering algo: {algo}")

        state.cluster_labels = np.asarray(labels)

        # Chunk-level purity against doc labels
        chunk_labels = [state.labels[c["doc_id"]] for c in state.chunks]
        state.metrics["purity"] = purity(chunk_labels, state.cluster_labels)
        state.metrics["n_clusters"] = int(len(set(state.cluster_labels)))
        return state