# CARE-LLM: Overcoming Data Quality and Structural Biases for Reliable Drug-Disease Interaction Prediction

CARE-LLM is a knowledge-graph-augmented LLM framework designed to improve drug–disease interaction prediction by addressing data quality issues and structural biases in biomedical knowledge graphs. It leverages a filtered, task-specific subgraph of DRKG to provide high-confidence, context-aware reasoning for LLMs.



## Requirements

- Python 3.9+
- PyTorch
- Transformers
- rdflib

## Dataset Access

You can download all required datasets and precomputed embeddings from the following link:

**[Download Dataset (Google Drive)]([PUT_YOUR_DRIVE_LINK_HERE](https://drive.google.com/file/d/1zW8sEGl3aGtwdIGzgAT18iw_PyE9rJPd/view?usp=sharing))**

Please download the archive and place the files in the appropriate folders as described below.

---

## Usage

1. **Prepare the datasets:**
   - Place the complete DRKG and its `Entity_embeddings.pkl` in:  
     `dataset/drkg-complet/`
   - Place the filtered subgraph (SubDRKG) and its corresponding `Entity_embeddings.pkl` in:  
     `dataset/subDRKG/`
   - Place the evaluation/test dataset and `keyword_embeddings.pkl` in:  
     `dataset/test/`

   **Note:** The folder `embedding/` contains `embedding.py`, which is responsible for generating embeddings for:
   - Knowledge graph entities, and  
   - Query entities (drug/disease/protein names)  
   
   These embeddings are used for similarity-based alignment.

2. **Run the main pipeline:**
```bash
python main.py
