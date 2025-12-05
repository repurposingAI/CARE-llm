import datasets
import random
random.seed(42)
import json
# from api_utils import *
import time
import os
import pdb
import csv
from rdflib import Graph, Namespace, URIRef
import json



class Processor:
    def __init__(self):
        self.template_ner = '''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
Question: {}
The extracted entities are:
'''
        self.template = '''Question: {} 
Answer: The option is: '''
        self.template_CoT = '''Question: {} 
Answer: Let's think step by step. '''
        self.template_inference = '''Question: {} 
Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''

    def load_dataset(self):
        return self.data

    def load_original_dataset(self):
        return self.data_original
    

class medmcqaZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medmcqa_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medmcqa_filter.json')))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        question = item['question']
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['opa'], item['opb'], item['opc'], item['opd']
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['cop']
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class medqaZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'medqa_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'medqa_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'medqa_filter.json')))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['answer'][0]
        answer = item['choices'].index(answer)
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class mmluZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'mmlu_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'mmlu_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'mmlu_filter.json')))
        self.num2answer = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

    def generate_prompt_ner(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner

    def generate_prompt(self, item):
        question = item['question']
        A, B, C, D = item['choices'][0], item['choices'][1], item['choices'][2], item['choices'][3]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['answer']
        answer = self.num2answer[answer]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc


class qa4mreZeroshotsProcessor(Processor):
    def __init__(self):
        super().__init__()
        if os.path.exists(os.path.join('Alzheimers','result_ner', 'qa4mre_zero-shot.json')):
            self.data = json.load(open(os.path.join('Alzheimers','result_ner', 'qa4mre_zero-shot.json')))
        self.data_original = json.load(open(os.path.join('Alzheimers', 'result_filter', 'qa4mre_filter.json')))
        self.num2answer = {
            1: 'A',
            2: 'B',
            3: 'C',
            4: 'D',
            5: 'E'
        }

    def generate_prompt_ner(self, item):
        question = item['question_str']
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option

        prompt_ner = self.template_ner.format(question)
        return prompt_ner


    def generate_prompt(self, item):
        question = item['question_str']
        A, B, C, D, E = item['answer_options']['answer_str'][0], item['answer_options']['answer_str'][1], item['answer_options']['answer_str'][2], item['answer_options']['answer_str'][3], item['answer_options']['answer_str'][4]
        option = '\n'+'A.'+A+'\n'+'B.'+B+'\n'+'C.'+C+'\n'+'D.'+D+'\n'+'E.'+E+'\n'
        question += option
        return question

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]
        item['prediction'] = ret
        answer = item['correct_answer_id']
        answer = self.num2answer[int(answer)]
        if answer.strip() == ret.strip():
            acc = 1
        else:
            acc = 0
        return item, acc





from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import re  
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep

#from dataset_utils import *
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import torch

from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pdb

# Collez ici votre clé d'accès Hugging Face
login("")

import torch

if torch.cuda.is_available():
    print("✅ GPU is available.")
    print("Using device:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU is NOT available.")


# Envoyer le modèle sur GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Charger le modèle SciBERT
scibert_name = "allenai/scibert_scivocab_uncased"
scibert_tokenizer = AutoTokenizer.from_pretrained(scibert_name)
scibert_model = AutoModel.from_pretrained(scibert_name)
scibert_model = scibert_model.to(device)
scibert_model.eval()





"""
max_memory_mapping = {
    0: "10GB",       # GPU (device 0)
    "cpu": "30GB"    # CPU
}

#llama3
#BioMistral-7B/
#meta-llama/Llama-2-7b-chat-hf
model_name = "/home/habes/NotebookProjects/bboi/sadok/BioMistral-7B/"  # ex. HuggingFaceHub/model‑ft‑medical
device_map = 'auto'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=True,
    torch_dtype="auto",  # Use appropriate precision 
    max_memory=max_memory_mapping
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.6,
    top_p=0.9,
    do_sample=True
)

chat = HuggingFacePipeline(pipeline=pipe)
"""

def load_llm(
    model_name="/home/habes/NotebookProjects/bboi/sadok/BioMistral-7B/",
    max_gpu_mem="10GB",
    max_cpu_mem="30GB",
):
    """
    Charge un modèle quantifié 4-bit + pipeline HuggingFace dans une seule fonction.
    Retourne l'objet chat = HuggingFacePipeline(...)
    """

    # Max memory mapping
    max_memory_mapping = {
        0: max_gpu_mem,  # GPU ID 0
        "cpu": max_cpu_mem
    }

    # Configuration quantization 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Charger modèle
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache=True,
        torch_dtype="auto",
        max_memory=max_memory_mapping,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Pipeline HF
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.6,
        top_p=0.9,
        do_sample=True
    )

    # LangChain wrapper
    chat = HuggingFacePipeline(pipeline=pipe)

    return chat



def load_cot(
    model_name="llama3",
    max_gpu_mem="10GB",
    max_cpu_mem="30GB",
):
    """
    Charge un modèle quantifié 4-bit + pipeline HuggingFace dans une seule fonction.
    Retourne l'objet chat = HuggingFacePipeline(...)
    """

    # Max memory mapping
    max_memory_mapping = {
        0: max_gpu_mem,  # GPU ID 0
        "cpu": max_cpu_mem
    }

    # Configuration quantization 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Charger modèle
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache=True,
        torch_dtype="auto",
        max_memory=max_memory_mapping,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Pipeline HF
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )

    # LangChain wrapper
    chat = HuggingFacePipeline(pipeline=pipe)

    return chat

from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import torch
import re

def rerank_knowledge_by_relevance(
    input_text: str,
    response_of_KG_list_path: str,
    response_of_KG_neighbor: str,
    sent_start: str,
    sent_end: str,
    tokenizer,
    model,
    entity_embeddings,
    top_k: int = 3,
    semantic_weight: float = 0.8,
) -> Tuple[str, str, str, str]:
    """
    Combine la similarité dense (SciBERT) avec un rerankeur CrossEncoder pour classer les connaissances.
    Rerank aussi les phrases (sent_start, sent_end).
    """

    cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base")

    def extract_entities(text):
        entities = []
        matches = re.findall(r"(?:'([^']+)'|(\b\w+::[^\s,]+))", text)
        for match in matches:
            entity = match[0] if match[0] else match[1]
            if entity and "::" in entity:
                entities.append(entity)
        return list(set(entities))

    def score_entity_similarity(entity_name):
        try:
            entity_idx = entity_embeddings["entities"].index(entity_name)
            entity_embedding = entity_embeddings["embeddings"][entity_idx]
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                question_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1).cpu().numpy()
            similarity = cosine_similarity([question_embedding[0]], [entity_embedding])[0][0]
            return similarity
        except:
            return 0.0

    def rerank(text_block: str) -> str:
        lines = [line.strip() for line in text_block.split("\n") if line.strip()]
        if not lines:
            return ""

        # Étape 1 : Similarité dense (SciBERT)
        line_scores: Dict[str, float] = {}
        for line in lines:
            entities = extract_entities(line)
            score = sum(score_entity_similarity(e) for e in entities)
            line_scores[line] = semantic_weight * score

        # Étape 2 : CrossEncoder reranking
        pairs = [(input_text, line) for line in lines]
        ce_scores = cross_encoder.predict(pairs)

        reranked = sorted(zip(lines, ce_scores), key=lambda x: x[1], reverse=True)
        top_passages = [line for line, score in reranked[:top_k]]

        return "\n".join(top_passages)

    # 🔹 Appliquer reranking aux trois sources de connaissances
    filtered_path = rerank(response_of_KG_list_path)
    filtered_neighbor = rerank(response_of_KG_neighbor)

    # 🔹 Rerank des phrases start/end
    sent_candidates = [sent_start, sent_end]
    pairs = [(input_text, s) for s in sent_candidates if s]
    ce_scores = cross_encoder.predict(pairs)
    reranked_sentences = [s for s, _ in sorted(zip(sent_candidates, ce_scores), key=lambda x: x[1], reverse=True)]

    # S'assurer qu'on retourne les deux dans le bon ordre (même si un seul est meilleur)
    new_sent_start = reranked_sentences[0] if len(reranked_sentences) > 0 else ""
    new_sent_end = reranked_sentences[1] if len(reranked_sentences) > 1 else ""

    return filtered_path, filtered_neighbor, new_sent_start, new_sent_end











def filter_knowledge_by_similarity(input_text, 
                                   response_of_KG_list_path, 
                                   response_of_KG_neighbor, 
                                   sent_start, 
                                   sent_end,
                                   tokenizer, 
                                   model, 
                                   entity_embeddings, 
                                   top_k=3):
    """
    Filtre les connaissances (path, neighbor, sent_start, sent_end)
    en gardant seulement les éléments les plus similaires à la question
    à l’aide des embeddings SciBERT et de la similarité cosinus.
    """

    device = model.device

    # === 1. Embedding de la question ===
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    question_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1).cpu().numpy()

    # === 2. Extraction des entités ===
    def extract_entities(text):
        entities = []
        matches = re.findall(r"(?:'([^']+)'|(\b\w+::[^\s,]+))", text)
        for match in matches:
            entity = match[0] if match[0] else match[1]
            if entity and "::" in entity:
                entities.append(entity)
        return list(set(entities))

    # === 3. Calcul de similarité entité-question ===
    def get_entity_similarity(entity_name):
        try:
            entity_idx = entity_embeddings["entities"].index(entity_name)
            entity_embedding = entity_embeddings["embeddings"][entity_idx]
            similarity = cosine_similarity([question_embedding], [entity_embedding])[0][0]
            return similarity
        except:
            return -1

    # === 4. Fonction de filtrage ===
    def filter_text(text, top_k):
        if not text:
            return ""
        entities = extract_entities(text)
        entity_scores = [(e, get_entity_similarity(e)) for e in entities]
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        top_entities = [e[0] for e in entity_scores[:top_k]]
        filtered_lines = [line for line in text.split("\n") if any(e in line for e in top_entities)]
        return "\n".join(filtered_lines) if filtered_lines else text.split("\n")[0]

    # === 5. Appliquer le filtrage ===
    filtered_path = filter_text(response_of_KG_list_path, top_k)
    filtered_neighbor = filter_text(response_of_KG_neighbor, top_k)
    filtered_sent_start = filter_text(sent_start, top_k)
    filtered_sent_end = filter_text(sent_end, top_k)

    return filtered_path, filtered_neighbor, filtered_sent_start, filtered_sent_end


def chat_response(prompt: str):
    return chat.invoke(prompt)


def prompt_extract_keyword(input_text):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay, Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat.invoke(chat_prompt_with_values.to_messages())


    question_kg = re.findall(re1,response_of_KG)
    return question_kg

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS
import re

# Créer un graphe RDF à partir de votre fichier TXT
def load_txt_to_rdf(txt_file_path):
    """
    Charge le fichier TXT et le convertit en graphe RDF
    """
    g = Graph()
    MY = Namespace("http://mybiogenont.org/ontology#")
    g.bind("my", MY)
    
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or '\t' not in line:
                continue
                
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
                
            subject, predicate_info, obj = parts
            
            # Extraire les informations de la prédicat
            pred_parts = predicate_info.split(' - ')
            if len(pred_parts) != 3:
                continue
                
            subj_type, pred_name, obj_type = pred_parts
            
            # Créer les URIs
            subj_uri = URIRef(MY[subject.replace(" ", "_")])
            obj_uri = URIRef(MY[obj.replace(" ", "_")])
            pred_uri = URIRef(MY[pred_name])
            
            # Ajouter les triplets au graphe
            g.add((subj_uri, RDF.type, URIRef(MY[subj_type])))
            g.add((obj_uri, RDF.type, URIRef(MY[obj_type])))
            g.add((subj_uri, pred_uri, obj_uri))
            
            # Ajouter les labels
            g.add((subj_uri, RDFS.label, Literal(subject)))
            g.add((obj_uri, RDFS.label, Literal(obj)))
    
    return g

# Charger le graphe RDF
#txt_file_path = "/kaggle/input/khalil2/relations_mapped.txt"
#rdf_graph = load_txt_to_rdf(txt_file_path)

import csv
import re
import pandas as pd

csv.field_size_limit(2**31 - 1)

def find_sentences_for_entities(entity1, entity2, csv_path):
    """
    Recherche les phrases pour chaque entité individuellement
    """
    sentences_entity1 = []
    sentences_entity2 = []
    
    # Nettoyer les noms d'entités
    entity1_clean = entity1.strip().lower()
    entity2_clean = entity2.strip().lower()
    
    try:
        # Utiliser pandas pour une lecture plus robuste
        df = pd.read_csv(csv_path)
        
        for index, row in df.iterrows():
            e1 = str(row["Entity1_name"]).strip().lower() if pd.notna(row["Entity1_name"]) else ""
            e2 = str(row["Entity2_name"]).strip().lower() if pd.notna(row["Entity2_name"]) else ""
            sent = str(row.get("Sentence_tokenized", "")).strip() if pd.notna(row.get("Sentence_tokenized")) else ""
            
            # Vérifier si la phrase n'est pas vide
            if not sent or sent == "nan" or sent == "":
                continue
            
            # Recherche pour entity1
            if entity1_clean in e1 or entity1_clean in e2:
                sentences_entity1.append(sent)
            
            # Recherche pour entity2  
            if entity2_clean in e1 or entity2_clean in e2:
                sentences_entity2.append(sent)
        
        # Prendre les premières phrases trouvées (ou toutes si vous voulez)
        sentence_start = " ".join(sentences_entity1[:3]) if sentences_entity1 else ""
        sentence_end = " ".join(sentences_entity2[:3]) if sentences_entity2 else ""
        
        print(f"Sentences trouvées pour {entity1}: {len(sentences_entity1)}")
        print(f"Sentences trouvées pour {entity2}: {len(sentences_entity2)}")
        
    except Exception as e:
        print(f"Erreur lecture CSV : {e}")
        sentence_start, sentence_end = "", ""
    
    return sentence_start, sentence_end

from rdflib import Graph, Namespace, URIRef
import re

EX = Namespace("http://example.org/")

def safe_uri(label):
    """Nettoie les entités pour produire une URI valide."""
    clean = label.strip()
    clean = clean.replace(" ", "_")      # remplace espaces
    clean = re.sub(r"[^a-zA-Z0-9_:.-]", "", clean)  # enlève caractères interdits
    return EX[clean]

def get_neighbors(graph, entity_name):
    """Retourne les voisins + relations d'une entité RDF."""
    
    # Enlever les apostrophes si présentes
    if entity_name.startswith("'") and entity_name.endswith("'"):
        clean_entity = entity_name[1:-1]  # Enlever les apostrophes
    elif entity_name.startswith('"') and entity_name.endswith('"'):
        clean_entity = entity_name[1:-1]  # Enlever les guillemets
    else:
        clean_entity = entity_name
    
 
    clean_entity = clean_entity.replace(" ", "_")

    query = f"""
    PREFIX ex: <http://example.org/>

    SELECT ?neighbor ?relation
    WHERE {{
        {{
            ex:{clean_entity} ?relation ?neighbor .
        }}
        UNION
        {{
            ?neighbor ?relation ex:{clean_entity} .
        }}
    }}
    """

    results = []

    for row in graph.query(query):
        neighbor = str(row.neighbor).split("/")[-1].replace("_", " ")
        relation = str(row.relation).split("/")[-1].replace("_", " ")
        
        # Formater comme vous le souhaitez
        formatted_result = {
            'entity': clean_entity.replace("_", " "),
            relation: neighbor
        }
        results.append(formatted_result)

    return results
    
def prompt_path_finding(path_input):
    chat = load_llm()
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.

    {Path}

    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...

    Output:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["Path"]
    )

    formatted_prompt = prompt.format(Path=path_input)

    # HuggingFacePipeline expects a string, not a list of messages.
    response_of_KG_path = chat.invoke(formatted_prompt)
    return response_of_KG_path


def extract_triplets_from_paths(path_text):
    triplets = []
    lines = path_text.strip().split("\n")

    for line in lines:
        # Ignorer "Chemin X (longueur Y): "
        if ":" in line:
            line = line.split(":", 1)[1].strip()

        # Découper en tokens
        tokens = [t.strip() for t in line.split("->")]

        # Extraire les triplets : (entity, relation, entity)
        for i in range(0, len(tokens) - 2, 2):
            e1 = tokens[i]
            rel = tokens[i+1]
            e2 = tokens[i+2]
            triplets.append(f"{e1} -> {rel} -> {e2}")

    return triplets


def prompt_paths(paths_text, question):
    chat = load_llm()
    # Extraire les triplets
    triplets = extract_triplets_from_paths(paths_text)

    if not triplets:
        print("Aucun triplet extrait depuis les paths")
        return "No relevant path-based evidence found."

    # Formatage pour le prompt
    graph_text = "\n".join(triplets)

    template = """
    There are some knowledge graph triples extracted from path-based reasoning in the format: entity->relationship->entity.

    {graph}

    Convert these triples to natural language evidence. Use single quotation marks for entity names and relation names.
    Name them as Path-based Evidence 1, Path-based Evidence 2, etc.

    Focus on drug-gene interactions and relationships that help answer the question.

    Question: {question}

    Output only the evidence in this format:
    Path-based Evidence 1: [natural language description]
    Path-based Evidence 2: [natural language description]
    Path-based Evidence 3: ...

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["graph", "question"]
    )

    formatted_prompt = prompt.format(graph=graph_text, question=question)
    response = chat.invoke(formatted_prompt)
    return response


def extract_reranked_triplets(reranked_output):
    """
    Extrait proprement les triplets rerankés du output
    """
    triplets = []
    
    # Diviser par lignes et chercher les triplets
    lines = reranked_output.split('\n')
    for line in lines:
        line = line.strip()
        
        # Chercher les lignes de triplets rerankés
        if (line.startswith('Reranked Triple') or 
            ('->' in line and not any(x in line for x in ['##', 'Question:', 'Answer:', 'Output:']))):
            
            # Extraire le triplet
            if ':' in line and line.startswith('Reranked Triple'):
                triplet = line.split(':', 1)[1].strip()
            else:
                triplet = line
            
            # Nettoyer le triplet
            triplet = triplet.replace(' ——> ', '->')
            triplet = triplet.replace(' -> ', '->')
            
            # Vérifier que c'est un vrai triplet (pas des xxx)
            if 'xxx' not in triplet and len(triplet.split('->')) == 3:
                triplets.append(triplet)
    
    return triplets

def prompt_neighbor(neighbor):
    """
    Version corrigée de prompt_neighbor
    """
    chat = load_llm()
 
    if isinstance(neighbor, str):
        triplets = extract_reranked_triplets(neighbor)
        
        if not triplets:
            print("Aucun triplet valide trouvé dans l'input")
       
            return "No relevant neighbor-based evidence found."
        
    
        neighbor_text = '\n'.join(triplets)
        print(f"Triplets extraits pour prompt_neighbor:")
        for triplet in triplets:
            print(f"   {triplet}")
    else:
        neighbor_text = str(neighbor)
    
    template = """
    There are some knowledge graph triples in the format: entity->relationship->entity

    {neighbor}

    Convert these triples to natural language evidence. Use single quotation marks for entity names and relation names. 
    Name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2, etc.

    Focus on drug-gene interactions and relationships that help answer the question.

    Output only the evidence in this format:
    Neighbor-based Evidence 1: [natural language description]
    Neighbor-based Evidence 2: [natural language description]
    ...

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["neighbor"]
    )

    try:
        formatted_prompt = prompt.format(neighbor=neighbor_text)
        response_of_KG_neighbor = chat.invoke(formatted_prompt)
        return response_of_KG_neighbor
    except Exception as e:
        print(f"Erreur dans prompt_neighbor: {e}")
        return "Error processing neighbor evidence."

        



def self_knowledge_retrieval(graph, question):
    chat = load_llm()
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.

    ##Graph: {graph}

    ##Question: {question}

    Please filter noisy knowledge from this knowledge graph that is useless or irrelevant to the given question. Output the filtered knowledges in the same format as the input knowledge graph.

    Filtered Knowledge:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["graph", "question"]
    )

    formatted_prompt = prompt.format(graph=graph, question=question)
    response = chat.invoke(formatted_prompt)
    return response

def self_knowledge_retrieval_reranking(neighbor_input, question):
    chatc = load_cot()
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.

    ##Graph: {graph}

    ##Question: {question}

    Please rerank the knowledge graph and output at most 5 important and relevant triples for solving the given question. Output the reranked knowledge in the following format:
    Reranked Triple1: xxx ——> xxx
    Reranked Triple2: xxx ——> xxx
    Reranked Triple3: xxx ——> xxx

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["graph", "question"]
    )

    formatted_prompt = prompt.format(graph=neighbor_input, question=question)
    response = chatc.invoke(formatted_prompt)
    return response

    

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def is_unable_to_answer(response):
    # À remplacer par un check basique ou ignorer cette fonction
    return False  # ou un scoring manuel si nécessaire



def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines


def final_answer(input_text, response_path_evidence, response_of_KG_neighbor):
    """
    Fonction généralisée pour déterminer l'interaction entre n'importe quel drug et target
    à partir des preuves de connaissance médicale.
    """

    chatc = load_cot()
    chat = load_llm()

    # Sécurité si vide
    response_path_evidence = response_path_evidence or ""
    response_of_KG_neighbor = response_of_KG_neighbor or ""

    # -----------------------------  
    # Extraction des Path Evidence
    # -----------------------------
    def extract_path_evidence(text):
        evidences = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Path-based Evidence"):
                evidences.append(line)
        return "\n".join(evidences)

    clean_path_evidence = extract_path_evidence(response_path_evidence)

    # -----------------------------  
    # Extraction des Neighbor Evidence
    # -----------------------------
    def extract_neighbor_evidence(text):
        evidences = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Neighbor-based Evidence"):
                evidences.append(line)
        return "\n".join(evidences) if evidences else text

    clean_neighbor_evidence = extract_neighbor_evidence(response_of_KG_neighbor)

    # Fusion
    all_evidence = clean_path_evidence + "\n" + clean_neighbor_evidence

    # -----------------------------  
    # PHASE 1 : Chain-of-Thought
    # -----------------------------
    prompt_cot = f"""
    You are a medical AI assistant answering a drug-target or drug-disease interaction question.

    QUESTION TO SOLVE:
    {input_text}

    MEDICAL KNOWLEDGE EVIDENCE:
    {all_evidence}

    Analyze the evidence step by step to determine whether an interaction exists.
    Your reasoning must:
    - Focus only on the entities mentioned in the question
    - Use only the medical evidence provided
    - Avoid any external assumptions

    Step-by-step reasoning:
    """

    result_CoT = chatc.invoke(prompt_cot)
    output_CoT = result_CoT

    # -----------------------------  
    # PHASE 2 : Final Answer (with protected few-shot)
    # -----------------------------
    prompt_final = f"""
    You are a medical AI assistant. Your goal is to answer the user's question clearly.

    === MEDICAL EVIDENCE ===
    {all_evidence}

    === REASONING SUMMARY ===
    {output_CoT}

    Using the reasoning above, provide the final answer.
    IMPORTANT:
    - Follow the format demonstrated in the few-shot examples.
    - DO NOT use their medical content.
    - DO NOT confuse the examples with the actual question.
    - They are ONLY formatting guides.

    Your answer MUST be exactly:
      Yes. <1-2 sentence rationale>
      or
      No. <1-2 sentence rationale>

    -----------------------------------------------------
    BEGIN TRAINING EXAMPLES (FORMAT ONLY — DO NOT COPY)
    -----------------------------------------------------

    Example 1:
    Question Example: Does Chemical and Drug Induced Liver Injury interact with Proguanil?
    Example Final Answer: Yes. Proguanil has been associated with cases of drug-induced liver injury in clinical reports. Although the reaction is uncommon,   
    hepatotoxicity is a documented adverse effect of Proguanil, indicating a plausible interaction between the drug and liver injury mechanisms.

    Example 2:
    Question Example: Does Frontometaphyseal dysplasia interact with Metamfetamine?
    Example Final Answer: Yes. Frontometaphyseal dysplasia involves multisystem developmental abnormalities, and patients may receive medications or           
    supportive treatments that can interact pharmacodynamically or pharmacokinetically with Metamfetamine. While the interaction is not directly documented, 
    the condition’s clinical management and comorbidities create a plausible context for potential interaction.
    

    -----------------------------------------------------
    END TRAINING EXAMPLES
    -----------------------------------------------------

    Now answer the REAL question:
    {input_text}

    Final Answer:
    """

    final_result = chat.invoke(prompt_final)
    return final_result

    



def prompt_document(question,instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content

    return response_document_bm25



def explore_similar_entities(rdf_graph, target_entities):
    """
    Explore le graphe pour trouver des entités similaires aux entités cibles
    """
    MY = Namespace("http://mybiogenont.org/ontology#")
    
    print(f"\n🔍 EXPLORATION DU GRAPHE POUR LES ENTITÉS: {target_entities}")
    

    all_entities = set()
    

    for triple in rdf_graph:
        subj = str(triple[0])
        obj = str(triple[2])
        
  
        if "http://mybiogenont.org/ontology#" in subj:
            subj_clean = subj.replace("http://mybiogenont.org/ontology#", "").replace("_", " ")
        elif "http://example.org/" in subj:
            subj_clean = subj.replace("http://example.org/", "").replace("_", " ")
        else:
            subj_clean = subj.split("/")[-1].replace("_", " ")
            
        if "http://mybiogenont.org/ontology#" in obj:
            obj_clean = obj.replace("http://mybiogenont.org/ontology#", "").replace("_", " ")
        elif "http://example.org/" in obj:
            obj_clean = obj.replace("http://example.org/", "").replace("_", " ")
        else:
            obj_clean = obj.split("/")[-1].replace("_", " ")
        
        all_entities.add(subj_clean)
        all_entities.add(obj_clean)
    
    print(f" Graphe contient {len(all_entities)} entités uniques")
    

    similar_entities = {}
    
    for target in target_entities:
        target_lower = target.lower()
        similar_entities[target] = []
        
        for entity in all_entities:
            entity_lower = entity.lower()
            
     
            if (target_lower in entity_lower or 
                entity_lower in target_lower or
                any(word in entity_lower for word in target_lower.split()) or
                any(word in target_lower for word in entity_lower.split())):
                
         
                entity_uri = entity.replace(" ", "_")
                count_query = f"""
                PREFIX my: <http://mybiogenont.org/ontology#>
                PREFIX ex: <http://example.org/>
                
                SELECT (COUNT(*) as ?count)
                WHERE {{
                    {{
                        {{ my:{entity_uri} ?p ?o }}
                        UNION
                        {{ ex:{entity_uri} ?p ?o }}
                    }}
                    UNION
                    {{
                        {{ ?s ?p my:{entity_uri} }}
                        UNION
                        {{ ?s ?p ex:{entity_uri} }}
                    }}
                }}
                """
                
                try:
                    count_result = list(rdf_graph.query(count_query))
                    triplet_count = int(count_result[0][0]) if count_result else 0
                    
                    similar_entities[target].append({
                        'entity': entity,
                        'triplet_count': triplet_count,
                        'similarity_score': calculate_similarity(target, entity)
                    })
                except Exception as e:
                    print(f"⚠️ Erreur comptage pour '{entity}': {e}")
                    similar_entities[target].append({
                        'entity': entity,
                        'triplet_count': 0,
                        'similarity_score': calculate_similarity(target, entity)
                    })
        
    
        similar_entities[target].sort(key=lambda x: (-x['similarity_score'], -x['triplet_count']))
    
    return similar_entities


def find_shortest_path_in_graph(rdf_graph, entity1, entity2, max_hops=3):
    """
    Trouve le plus court chemin entre deux entités dans le graphe RDF
    """
    print(f" Searching for paths between: '{entity1}' et '{entity2}'")
    

    entity1_clean = entity1.replace(" ", "_")
    entity2_clean = entity2.replace(" ", "_")
    

    query_paths = f"""
    PREFIX my: <http://mybiogenont.org/ontology#>
    PREFIX ex: <http://example.org/>

    SELECT ?pathLength ?rel1 ?inter1 ?rel2 ?inter2 ?rel3
    WHERE {{
        {{
            # Chemins directs (1 saut) - essayer les deux namespaces
            {{
                my:{entity1_clean} ?rel1 my:{entity2_clean} .
            }}
            UNION
            {{
                ex:{entity1_clean} ?rel1 ex:{entity2_clean} .
            }}
            UNION
            {{
                my:{entity1_clean} ?rel1 ex:{entity2_clean} .
            }}
            UNION
            {{
                ex:{entity1_clean} ?rel1 my:{entity2_clean} .
            }}
            BIND(1 as ?pathLength)
        }}
        UNION
        {{
            # Chemins avec 1 intermédiaire (2 sauts)
            {{
                my:{entity1_clean} ?rel1 ?inter1 .
                ?inter1 ?rel2 my:{entity2_clean} .
            }}
            UNION
            {{
                ex:{entity1_clean} ?rel1 ?inter1 .
                ?inter1 ?rel2 ex:{entity2_clean} .
            }}
            BIND(2 as ?pathLength)
        }}
        UNION
        {{
            # Chemins avec 2 intermédiaires (3 sauts)
            {{
                my:{entity1_clean} ?rel1 ?inter1 .
                ?inter1 ?rel2 ?inter2 .
                ?inter2 ?rel3 my:{entity2_clean} .
            }}
            UNION
            {{
                ex:{entity1_clean} ?rel1 ?inter1 .
                ?inter1 ?rel2 ?inter2 .
                ?inter2 ?rel3 ex:{entity2_clean} .
            }}
            BIND(3 as ?pathLength)
        }}
    }}
    ORDER BY ?pathLength
    LIMIT 20
    """
    
    paths = []
    
    try:
        results = list(rdf_graph.query(query_paths))
        print(f"🔍 {len(results)} résultat(s) SPARQL trouvé(s)")
        
        for row in results:
            path_length = int(row.pathLength)
            
        
            def clean_uri(uri):
                uri_str = str(uri)
                if "http://mybiogenont.org/ontology#" in uri_str:
                    return uri_str.replace("http://mybiogenont.org/ontology#", "").replace("_", " ")
                elif "http://example.org/" in uri_str:
                    return uri_str.replace("http://example.org/", "").replace("_", " ")
                else:
                    return uri_str.split("/")[-1].replace("_", " ")
            
            path_info = {
                'length': path_length,
                'entities': [entity1],
                'relations': [],
                'raw_data': []
            }
            
            if path_length == 1:
                rel1 = clean_uri(row.rel1)
                path_info['entities'].append(entity2)
                path_info['relations'].append(rel1)
                path_info['raw_data'].append({
                    'from': entity1, 'rel': rel1, 'to': entity2
                })
                
            elif path_length == 2:
                rel1 = clean_uri(row.rel1)
                rel2 = clean_uri(row.rel2)
                inter1 = clean_uri(row.inter1)
                path_info['entities'].extend([inter1, entity2])
                path_info['relations'].extend([rel1, rel2])
                path_info['raw_data'].extend([
                    {'from': entity1, 'rel': rel1, 'to': inter1},
                    {'from': inter1, 'rel': rel2, 'to': entity2}
                ])
                
            elif path_length == 3:
                rel1 = clean_uri(row.rel1)
                rel2 = clean_uri(row.rel2)
                rel3 = clean_uri(row.rel3)
                inter1 = clean_uri(row.inter1)
                inter2 = clean_uri(row.inter2)
                path_info['entities'].extend([inter1, inter2, entity2])
                path_info['relations'].extend([rel1, rel2, rel3])
                path_info['raw_data'].extend([
                    {'from': entity1, 'rel': rel1, 'to': inter1},
                    {'from': inter1, 'rel': rel2, 'to': inter2},
                    {'from': inter2, 'rel': rel3, 'to': entity2}
                ])
            
            paths.append(path_info)
            
    
        paths.sort(key=lambda x: x['length'])
        
    except Exception as e:
        print(f" Erreur SPARQL: {e}")
        import traceback
        traceback.print_exc()
    
    return paths

def calculate_similarity(str1, str2):
    """Calcule une similarité simple entre deux chaînes"""
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
 
    if str1_lower == str2_lower:
        return 1.0
    elif str1_lower in str2_lower or str2_lower in str1_lower:
        return 0.8
    else:
 
        words1 = set(str1_lower.split())
        words2 = set(str2_lower.split())
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))

def find_best_entity_matches(rdf_graph, entity1, entity2):

    similar_entities = explore_similar_entities(rdf_graph, [entity1, entity2])
    
    best_matches = {}
    
    for target, matches in similar_entities.items():
        print(f"\n MEILLEURES CORRESPONDANCES POUR '{target}':")
        
        if matches:
            for i, match in enumerate(matches[:5]):  # Top 5
                print(f"   {i+1}. '{match['entity']}' (score: {match['similarity_score']:.2f}, triplets: {match['triplet_count']})")
            
            best_matches[target] = matches[0]['entity']
        else:
            print(f"   Aucune correspondance trouvée pour '{target}'")
            best_matches[target] = None
    
    return best_matches


def find_entity_neighbors(rdf_graph, entity_name, limit=10):

    entity_uri = entity_name.replace(" ", "_")
    

    query = f"""
    PREFIX my: <http://mybiogenont.org/ontology#>
    PREFIX ex: <http://example.org/>

    SELECT ?relation ?neighbor ?direction
    WHERE {{
        {{
            # Relations sortantes - essayer les deux namespaces
            {{
                my:{entity_uri} ?relation ?neighbor .
            }}
            UNION
            {{
                ex:{entity_uri} ?relation ?neighbor .
            }}
            BIND("outgoing" as ?direction)
        }}
        UNION
        {{
            # Relations entrantes - essayer les deux namespaces
            {{
                ?neighbor ?relation my:{entity_uri} .
            }}
            UNION
            {{
                ?neighbor ?relation ex:{entity_uri} .
            }}
            BIND("incoming" as ?direction)
        }}
    }}
    LIMIT {limit}
    """
    
    neighbors = []
    
    try:
        results = rdf_graph.query(query)
        for row in results:
    
            def clean_uri(uri):
                uri_str = str(uri)
                if "http://mybiogenont.org/ontology#" in uri_str:
                    return uri_str.replace("http://mybiogenont.org/ontology#", "").replace("_", " ")
                elif "http://example.org/" in uri_str:
                    return uri_str.replace("http://example.org/", "").replace("_", " ")
                else:
                    return uri_str.split("/")[-1].replace("_", " ")
            
            relation = clean_uri(row.relation)
            neighbor = clean_uri(row.neighbor)
            direction = str(row.direction)
            
            neighbors.append({
                'relation': relation,
                'neighbor': neighbor,
                'direction': direction
            })
    except Exception as e:
        print(f" Erreur recherche voisins: {e}")
    
    return neighbors

import re

def clean_uri(value):

    if value is None:
        return "UNKNOWN"

    value = str(value).strip()


    value = value.replace("->", "_")
    value = value.replace("=>", "_")


    value = value.replace(" ", "_")


    value = re.sub(r"[^A-Za-z0-9_]", "_", value)


    if value == "":
        return "EMPTY"

    return value




if __name__ == "__main__":

    rdf_graph = load_txt_to_rdf("/home/habes/NotebookProjects/bboi/sadok/GNN-LLM/data/dalk/datasets/DRKG/DRKG_final.txt")


    chatc = load_cot()
    chat = load_llm()

    with open('/home/habes/NotebookProjects/bboi/sadok/GNN-LLM/data/dalk/datasets/DRKG/DRKG_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open('/home/habes/NotebookProjects/bboi/sadok/GNN-LLM/data/dalk/datasets/test/keyword_embeddings.pkl', 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    output_path = '/home/habes/NotebookProjects/bboi/sadok/GNN-LLM/data/dalk/result/cold_drug-t6-DRKGcomplet-few.jsonl'
    input_path = "/home/habes/NotebookProjects/bboi/sadok/GNN-LLM/data/dalk/datasets/test/cold_drug.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue

        try:
            drug, go_term = parts
            question = f"Does {drug} interacts with {go_term}:\nA.yes \nB.no\n"
            input_text = [question]
            entity_list = [f"1. {drug}, 2. {go_term}"]

            print(f"\n Processing question: {question}")

 
            question_kg = []
            entity1 = entity_list[0].split(',')[0].split('1.')[1].strip()
            entity2 = entity_list[0].split(',')[1].split('2.')[1].strip()
            question_kg.append([entity1, entity2])

     
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
   

            keyword_to_index = {kw: i for i, kw in enumerate(keyword_embeddings["keyword"])}


            entity_emb_matrix = np.array(entity_embeddings_emb)
            entity_emb_norm = entity_emb_matrix / np.linalg.norm(entity_emb_matrix, axis=1, keepdims=True)

            # ---------------------------------------------------------

            match_kg = []

            for kg_entity_pair in question_kg:
                pair_matches = []

                for kg_entity in kg_entity_pair:
              
                    kw_index = keyword_to_index[kg_entity]

            
                    kg_emb = keyword_embeddings["embeddings"][kw_index]
                    kg_emb = kg_emb / np.linalg.norm(kg_emb)

          
                    cos_sim = entity_emb_norm @ kg_emb

             
                    best_idx = np.argmax(cos_sim)
                    best_match = entity_embeddings["entities"][best_idx]

         
                    while best_match in pair_matches:
                        cos_sim[best_idx] = -1
                        best_idx = np.argmax(cos_sim)
                        best_match = entity_embeddings["entities"][best_idx]

                    pair_matches.append(best_match)

                match_kg.append(pair_matches)
              
            first_item = match_kg[0][0]


            EX = Namespace("http://example.org/")

            rdf_graph = Graph()

            with open("/home/habes/NotebookProjects/bboi/sadok/GNN-LLM/data/dalk/datasets/DRKG/DRKG_final.txt", "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 3:
                        continue

                    s = row[0]  
                    p = row[1]
                    o = row[2]

                    subj = URIRef(EX[clean_uri(s)])
                    pred = URIRef(EX[clean_uri(p)])
                    obj  = URIRef(EX[clean_uri(o)])

                    rdf_graph.add((subj, pred, obj))

            print("RDF loaded:", len(rdf_graph), "triples")

       
            entity = first_item  
            neighbors = get_neighbors(rdf_graph, entity)
            print("\n Neighbors returned:")
            print(neighbors)


          


            entity1 = match_kg[0][0]   # salmeterol
            entity2 = match_kg[0][1]   # adrb2
      

            print(f"\n🎯 A SIMPLIFIED APPROACH TO: '{entity1}' ET '{entity2}'")


            print(f"\n🔍 ENTITIES TO USE:")
            print(f"   '{entity1}' → '{entity1}'")
            print(f"   '{entity2}' → '{entity2}'")

        
            print(f"\n🔍 DIRECT ROUTE SEARCH...")
            paths = find_shortest_path_in_graph(rdf_graph, entity1, entity2, max_hops=3)
            if paths:
                print(f"\n {len(paths)} path found")
                for i, path in enumerate(paths):
                    display = f"Chemin {i+1} (length{path['length']}): "
                    entities = path['entities']
                    relations = path['relations']
        
                    for j in range(len(relations)):
                        display += f"{entities[j]} --[{relations[j]}]--> "
                    display += entities[-1]
                    print(f"   {display}")
    
                result_path = paths[0]
            else:
                print("NO PATH FOUND")
                result_path = None                    

           
    
            if len(match_kg) >= 2:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    result_new_path = ["->".join(p) for p in result_path]
                    path = "\n".join(result_new_path)
                    path_sampled = self_knowledge_retrieval_reranking(path, input_text[0])
                    response_of_KG_list_path = prompt_path_finding(path_sampled)
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path_sampled)
            else:
                response_of_KG_list_path = '{}'

         
            graph_lines = []

            for i, p in enumerate(paths, start=1):
                entities = p["entities"]
                relations = p["relations"]
                length = p["length"]

                line = f"path {i} (length {length}): "
                for j in range(len(relations)):
                    line += f"{entities[j]} -> {relations[j]} -> "
                line += entities[-1]

                graph_lines.append(line)

          
            path = "\n".join(graph_lines)
            

            response_of_KG_list_path = self_knowledge_retrieval_reranking(path, input_text[0])
            response_path_evidence = prompt_paths(response_of_KG_list_path, input_text[0])
            
         
            neighbor_new_list = []
            for neighbor_dict in neighbors:
                entity = neighbor_dict['entity']
            
                relation_key = [k for k in neighbor_dict.keys() if k != 'entity'][0]
                neighbor_entity = neighbor_dict[relation_key]
                neighbor_new_list.append(f"{entity}->{relation_key}->{neighbor_entity}")            
      
            neighbor_input = "\n".join(neighbor_new_list[:5]) if len(neighbor_new_list) > 5 else "\n".join(neighbor_new_list)
            neighbor_input_sampled = self_knowledge_retrieval_reranking(neighbor_input, input_text[0])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
                

  
            output_all = final_answer(input_text[0], response_path_evidence, response_of_KG_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text[0], response_path_evidence, response_of_KG_neighbor)
     
            match = re.search(r'(the answer is ([A-D])\.)', output_all.lower())
            if not match:
                match = re.search(r'(\n    ([A-D])\.)', output_all)
            predicted_answer = match.group(2) if match else "Not found"

            ret_parsed = {
                "input": input_text[0],
                "answer": predicted_answer,
                "output": output_all,
                "result_path": response_path_evidence,
                "response_of_KG_neighbor": response_of_KG_neighbor
            }

            with open(output_path, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(ret_parsed, ensure_ascii=False) + "\n")

            print(f" Result saved for the couple : {drug} - {go_term}")

        except Exception as e:
            print(f" Error processing the couple {line.strip()} : {e}")
            continue