import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from core.config import ROOT, flatten, load_config
from core.seeding import set_seed
from core.state import PipelineState
from pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to an ablation YAML")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    df = pd.read_parquet(ROOT / cfg["data"]["path"])
    if cfg["data"].get("sample_size"):
        n = min(int(cfg["data"]["sample_size"]), len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    state = PipelineState(config=cfg, df=df)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=args.run_name or Path(args.config).stem):
        mlflow.log_params(flatten(cfg))
        state = run_pipeline(state)
        mlflow.log_metrics({k: v for k, v in state.metrics.items() if isinstance(v, (int, float))})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summaries.json"
            out.write_text(json.dumps(state.summaries or [], indent=2))
            mlflow.log_artifact(str(out))

    print(json.dumps(state.metrics, indent=2))


if __name__ == "__main__":
    main()