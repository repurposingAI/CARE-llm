# CARE-LLM: Overcoming Data Quality and Structural Biases for Reliable Drug-Disease Interaction Prediction

CARE-LLM is a knowledge-graph-augmented LLM framework designed to improve drug–disease interaction prediction by addressing data quality issues and structural biases in biomedical knowledge graphs. It leverages a filtered, task-specific subgraph of DRKG to provide high-confidence, context-aware reasoning for LLMs.

## Overall framework

![description](https://github.com/repurposingAI/CARE-llm/blob/main/DTI.png)



## Requirements

###For Running main.py

- Python 3.9+
- PyTorch
- Transformers
- rdflib

## Dataset Access

You can download all required datasets and precomputed embeddings from the following link:

  **[Download Dataset (Google Drive)](https://drive.google.com/file/d/1zW8sEGl3aGtwdIGzgAT18iw_PyE9rJPd/view?usp=sharing)**


Please download the archive and place the files in the appropriate folders as described below.

---

## Usage

1. **Prepare the datasets:**
   - Place the complete DRKG (in a .txt file) and its `Entity_embeddings.pkl` in:  
     `dataset/DRKG/`
   - Place the filtered subgraph (SubDRKG) and its corresponding `Entity_embeddings.pkl` in:  
     `dataset/SUBDRKG/`
   - Place the evaluation/test dataset and `keyword_embeddings.pkl` in:  
     `dataset/test/`

   **Note:** The folder `DRKG-embedding/` contains `entity-embedding-tsv.py`, which is responsible for generating embeddings for:
   - Knowledge graph entities.
   - Query entities (drug/disease/protein names).
   
   These embeddings are used for similarity-based alignment.

2. **Run the main pipeline:**
```bash
python main.py


# Results: Impact of DRKG Filtering on CARE-LLM Performance

This section presents the accuracy of CARE-LLM for cold drug and cold disease predictions using:
- the full DRKG (unfiltered knowledge graph)
- the filtered DRKG (knowledge graph reduced to task-relevant biomedical entities)

Evaluations are performed under different LLM sampling temperatures (0.1, 0.3, 0.6).

## Understanding the Temperature of the LLM
The temperature parameter controls the randomness of the LLM:
- Low temperature (0.1) produces stable and deterministic predictions.
- Medium temperature (0.3) introduces moderate variability.
- High temperature (0.6) generates more diverse outputs but decreases accuracy.

Lower temperatures generally give more reliable predictions in biomedical settings.

## DRKG vs. Filtered DRKG

### DRKG (full graph)
The full DRKG contains many heterogeneous biomedical entities, including several that are not directly relevant to drug-target or drug–disease prediction.  
This increases noise and leads to lower overall accuracy, especially when the LLM operates at higher temperatures.

### Filtered DRKG
The filtered version retains only entities directly relevant to the task (drug, disease, protein, gene, and curated relations).  
By removing noisy or irrelevant nodes, the model produces higher accuracy and more stable predictions.

## Accuracy Table

| Setting        | Temp (0.1) | Temp (0.3) | Temp (0.6) |
|----------------|------------|------------|------------|
| **DRKG**       |            |            |            |
| Cold Drug      | 0.75       | 0.63       | 0.50       |
| Cold Disease   | 0.94       | 0.85       | 0.79       |
| **Filtered DRKG** |        |            |            |
| Cold Drug      | 0.84       | 0.79       | 0.59       |
| Cold Disease   | 0.97       | 0.95       | 0.775      |
