# CARE-LLM: Overcoming Data Quality and Structural Biases for Reliable Drug-Disease Interaction Prediction

CARE-LLM is a knowledge-graph-augmented LLM framework designed to improve drug–disease interaction prediction by addressing data quality issues and structural biases in biomedical knowledge graphs. It leverages a filtered, task-specific subgraph of DRKG to provide high-confidence, context-aware reasoning for LLMs.

## Overall framework

![description](https://github.com/repurposingAI/CARE-llm/blob/main/DTI.png)



## Requirements

### For Running main.py
To run the Drug Disease interaction prediction pipeline, please ensure the following Python dependencies are installed:

```
- Python 3.9+
- accelerate 0.27.0
- gensim 4.3.1
- huggingface-hub 0.36.0
- langchain 0.0.354
- langchain-community 0.0.20
- langchain-core 0.1.23
- numpy 1.26.4
- Pillow 9.5.0
- rdflib 7.2.1
- scipy 1.10.1
- sentence-transformers 2.7.0
- sentencepiece 0.1.99
- torch 2.9.1
- transformers 4.57.3
```

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

### Installation Guide

To get started with our approach CARE-llm, follow the steps below. The execution was tested on a uCloud server running Linux and requires at least 32 GB of available RAM.

1. Clone the repository

```
git clone https://github.com/repurposingAI/CARE-llm.git
cd CARE-llm
```

2. Install Python dependencies

Before running the code, make sure to install all required libraries listed in the Requirements section. Once these are installed, proceed with the main execution.

3. Run the prediction script

After setting up the environment and dependencies, you can directly launch the Drug Disease interaction prediction task using:

```
python3 main.py
```

⚠️ Make sure you have added your Hugging Face access token, and obtained permission to use specific model (BioMistral/BioMistral-7B). See the section Authentication & Model Access Setup for more details.

## The process of fine-tuning our large language model

The fine_tuning directory contains the script alpaca.py, which provides the code for performing Alpaca-style fine-tuning on the model BioMistral/BioMistral-7B. This fine-tuning process follows the self-instruct methodology to adapt the base model to biomedical question-answering tasks. In addition, the directory includes the notebook chain-of-thought-ft.ipynb, which implements a fine-tuning pipeline for the model meta-llama/Llama-3.3-70B-Instruct, specifically designed to incorporate chain-of-thought reasoning during training.

## Authentication & Model Access Setup

To run the Drug Disease interaction prediction (main.py) script, follow these steps to authenticate and gain access to the required LLMs:

Hugging Face Token Access (for BioMistral models)
To access BioMistral/BioMistral-7B:

Create a Hugging Face account at https://huggingface.co

Go to your settings → Access Tokens and generate a token

Important: You must request access to the following models:

BioMistral/BioMistral-7B

Once access is granted, you must insert your Hugging Face token into the file main.py by modifying the following code:

```
from huggingface_hub import login
login("your_token")
```

| Cold Drug      | 0.84       | 0.79       | 0.59       |
| Cold Disease   | 0.97       | 0.95       | 0.775      |
