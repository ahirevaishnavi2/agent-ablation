# Agentic NLP Clustering & Ablation Pipeline

##  Overview
This project is a modular, multi-agent Natural Language Processing (NLP) pipeline designed to ingest large volumes of unstructured text, automatically cluster them into distinct topics, and generate extractive summaries for each cluster.

Using the **20 Newsgroups dataset** (20,000+ raw forum posts), this project moves beyond a basic ML script by treating the NLP process as an **assembly line of independent agents**. More importantly, it implements a rigorous **systematic ablation study** using **MLflow** to scientifically prove which model configurations yield the highest accuracy with the lowest latency.

##  The Problem
Organizations and researchers possess massive amounts of unstructured text. 
1. Humans cannot manually read and organize thousands of documents.
2. Standard ML models require specific configurations (chunk size, embedding model, cluster count) which are usually chosen by intuition, leading to suboptimal accuracy and high compute costs.

##  The Solution
We built a YAML-driven pipeline where every stage is isolated into an "Agent." This allows us to run **Ablation Studies**—changing exactly *one* variable at a time—and track the metrics using MLflow to find the perfect "recipe" for the pipeline. 

### The 6 Agents:
1. **Preprocess Agent:** Cleans the text, removes binary gibberish, and drops empty rows.
2. **Chunking Agent:** Splits long documents into smaller, embeddable windows.
3. **Embedding Agent:** Uses HuggingFace `Sentence-Transformers` to convert text chunks into numerical vectors.
4. **Clustering Agent:** Groups the vectors using `K-Means` or `HDBSCAN`.
5. **Summarizer Agent:** Uses frequency-based sentence scoring to extract the most important sentences from each cluster.
6. **Critic Agent:** Calculates evaluation metrics (Purity, compression ratio, etc.) and logs them.

---

## 🛠️ Technology Stack
*   **Core NLP & ML:** Python, Scikit-learn, HuggingFace (`sentence-transformers`)
*   **Models Evaluated:** `all-MiniLM-L6-v2` (Fast/384d) vs. `all-mpnet-base-v2` (Accurate/768d)
*   **MLOps & Tracking:** MLflow, PyYAML
*   **Data Processing:** Pandas, NumPy

---

##  Ablation Study Results (The MLflow Experiment)
By integrating MLflow, we tracked the trade-offs between accuracy (Purity) and computational cost (Time). 

| Run Name | Chunk Size | Clusters (K) | Embedding Model | Purity Score | Embedding Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_kmeans_15` | 256 | 15 | MiniLM-L6 | 0.519 | ~8 mins |
| `baseline_kmeans_20` | 256 | 20 | MiniLM-L6 | 0.565 | ~5 mins |
| `chunk_size_128` | 128 | 20 | MiniLM-L6 | 0.561 | ~7 mins |
| `clustering_hdbscan` | 256 | Auto (4) | MiniLM-L6 | 0.138 | ~4 mins |
| `embed_mpnet_base` | 256 | 20 | MPNet-Base | **0.578** | **~1.2 hours** |

**Key Insight:** While the heavier `all-mpnet-base-v2` model achieved the highest purity (0.578), the lighter `all-MiniLM-L6-v2` achieved near-identical results (0.565) at **14x the speed**, making it the optimal choice for production environments. Furthermore, `K-Means` drastically outperformed `HDBSCAN` for high-dimensional text clustering.

### 📸 Output Screenshots
*(Replace the links below with actual image paths from your GitHub repo)*

#### 1. MLflow Experiment Tracking Dashboard
*Shows the comparison of all ablation runs, tracking Purity vs. Time.*
<img width="1710" height="605" alt="Screenshot 2026-11-14 154154" src="https://github.com/user-attachments/assets/fe84609e-759b-4242-88b6-00676abb81ba" />


#### 2. Generated Summaries (Extractive)
*Shows the output of the Summarizer Agent, cleanly grouping 1990s forum posts into readable topics.*
```json
[
  {
    "cluster": 1,
    "size": 216,
    "summary": "The Clipper Chip contains a classified single-key 64-bit block encryption algorithm called Skipjack. The algorithm uses 80 bit keys and has 32 rounds of scrambling."
  },
  {
    "cluster": 15,
    "size": 287,
    "summary": "Vancouver: 4 Power play: 6-1 Scorer Vancouver, Adams 1 (Linden, Bure) pp, 1:23."
  }
]



```
## 🚀 How to Run

### 1. Installation
```bash
git clone https://github.com/your-username/agent-ablation.git
cd agent-ablation
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```
### 2. Start MLFlow server
(Navigate to http://localhost:5000 in your browser)
```bash
mlflow ui
```
### 3. Run an Ablation Experiment
In your main terminal, run the pipeline using a specific YAML config:
```bash
# Baseline run
export PYTHONPATH="src"
python src/run_ablation.py --config configs/ablations/baseline_kmeans_20.yaml --run-name baseline_kmeans_k20
```


