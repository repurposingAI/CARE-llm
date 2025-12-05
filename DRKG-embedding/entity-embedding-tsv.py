import os
import json
import pandas as pd
from tqdm import tqdm
import torch
import pickle
from transformers import BertTokenizer, BertModel
from huggingface_hub import login

# --- Login HuggingFace ---
login("")

# --- Load SciBERT ---
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
model = BertModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------------------------------
#           FUNCTION : CLEAN ENTITY STRING
# ---------------------------------------------------------
def clean_entity(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ["nan", "none"]:
        return None

    # Remove strange unicode chars
    x = x.replace("\x00", "").replace("\u0000", "")
    return x

# ---------------------------------------------------------
#             EMBEDDING FUNCTION (robust)
# ---------------------------------------------------------
def embed_texts(texts, model, tokenizer, batch_size=64):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]

            try:
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=128
                )
            # If tokenizer crashes for a weird batch, skip it
            except Exception as e:
                print(f"[⚠️ SKIPPED BATCH] {batch} → error : {e}")
                continue

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

            all_embeddings.append(cls_embeddings.cpu())

    if len(all_embeddings) == 0:
        raise ValueError("❌ Aucune embedding générée ! Vérifie ton dataset.")
    return torch.cat(all_embeddings, dim=0).numpy()


# ---------------------------------------------------------
#            LOAD TSV AND EXTRACT CLEAN ENTITIES
# ---------------------------------------------------------
df = pd.read_csv("DRKG_final.tsv", sep="\t", header=None, names=["head", "relation", "tail"])
print("Nombre de lignes :", len(df))

entities = set()

for _, row in tqdm(df.iterrows(), total=len(df)):
    h = clean_entity(row["head"])
    t = clean_entity(row["tail"])
    if h: entities.add(h)
    if t: entities.add(t)

entities = list(entities)
print("Nombre d'entités uniques nettoyées :", len(entities))

# ---------------------------------------------------------
#                  GENERATE EMBEDDINGS
# ---------------------------------------------------------
entity_embeddings = embed_texts(entities, model, tokenizer)

# ---------------------------------------------------------
#                  SAVE RESULTS
# ---------------------------------------------------------
output = {
    "entities": entities,
    "embeddings": entity_embeddings,
}

with open("DRKG_embeddings.pkl", "wb") as f:
    pickle.dump(output, f)

print("✔ Embeddings générés et sauvegardés dans DRKG_embeddings.pkl")
