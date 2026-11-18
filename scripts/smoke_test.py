import pandas as pd

from core.config import load_config
from core.seeding import set_seed
from core.state import PipelineState
from pipeline import run_pipeline


def main():
    print("1. Loading config and setting seed...")
    cfg = load_config()
    set_seed(42)
    
    print("2. Loading data from parquet...")
    df = pd.read_parquet("data/processed/sample_v1.parquet")
    
    # Take a small slice just to test that all agents play nicely together
    df_small = df.sample(n=500, random_state=42).copy()
    
    print(f"3. Running pipeline on {len(df_small)} rows...")
    state = PipelineState(config=cfg, df=df_small)
    
    # This executes preprocess -> chunking -> embedding -> clustering -> summarizer -> critic
    state = run_pipeline(state)
    
    print("\n--- Smoke Test Results ---")
    for k, v in state.metrics.items():
        print(f"{k}: {v}")
        
    # Print a sample summary to verify the extractive summarizer is working
    if state.summaries:
        print("\n--- Sample Summary (Cluster 0) ---")
        print(state.summaries[0]["summary"][:300] + "...")
    else:
        print("\nNo summaries generated.")

if __name__ == "__main__":
    main()