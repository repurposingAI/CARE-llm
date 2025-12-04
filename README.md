# CARE-LLM: Overcoming Data Quality and Structural Biases for Reliable Drug-Disease Interaction Prediction

CARE-LLM is a knowledge-graph-augmented LLM framework designed to improve drug–disease interaction prediction by addressing data quality issues and structural biases in biomedical knowledge graphs. It leverages a filtered, task-specific subgraph of DRKG to provide high-confidence, context-aware reasoning for LLMs.



## Requirements

- Python 3.9+
- PyTorch
- Transformers
- rdflib

## Usage

1. Prepare the datasets:
   - Place the complete DRKG Entity_embeddings.pkl in `dataset/drkg-complet/`.
   - Place the filtered subgraph and Entity_embeddings.pkl in `dataset/subDRKG/`.
   - Place the evaluation/test dataset and keyword_embeddings.pkl in `dataset/test/`.

2. Run the main pipeline:
```bash
python main.py
