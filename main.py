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





#from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
#from langchain_core.prompts import PromptTemplate
#from langchain.chains import LLMChain
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
#from neo4j import GraphDatabase, basic_auth
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
#from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep

#from dataset_utils import *
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# Hugging Face authentication is picked up from the HF_TOKEN environment variable.
# Never store access tokens in source code.

import torch

if torch.cuda.is_available():
    print("âœ… GPU is available.")
    print("Using device:", torch.cuda.get_device_name(0))
else:
    print("âŒ GPU is NOT available.")


# Envoyer le modÃ¨le sur GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Charger le modÃ¨le SciBERT
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
model_name = "/home/habes/NotebookProjects/bboi/sadok/BioMistral-7B/"  # ex. HuggingFaceHub/modelâ€‘ftâ€‘medical
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
    model_name="BioMistral-7B/",
    max_gpu_mem="10GB",
    max_cpu_mem="30GB",
):
    """
    Charge un modÃ¨le quantifiÃ© 4-bit + pipeline HuggingFace dans une seule fonction.
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

    # Charger modÃ¨le
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
        temperature=0.1,
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
    Charge un modÃ¨le quantifiÃ© 4-bit + pipeline HuggingFace dans une seule fonction.
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

    # Charger modÃ¨le
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

#chat_bio = load_llm()
#chat_llama = load_cot()

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
    Combine la similaritÃ© dense (SciBERT) avec un rerankeur CrossEncoder pour classer les connaissances.
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

        # Ã‰tape 1 : SimilaritÃ© dense (SciBERT)
        line_scores: Dict[str, float] = {}
        for line in lines:
            entities = extract_entities(line)
            score = sum(score_entity_similarity(e) for e in entities)
            line_scores[line] = semantic_weight * score

        # Ã‰tape 2 : CrossEncoder reranking
        pairs = [(input_text, line) for line in lines]
        ce_scores = cross_encoder.predict(pairs)

        reranked = sorted(zip(lines, ce_scores), key=lambda x: x[1], reverse=True)
        top_passages = [line for line, score in reranked[:top_k]]

        return "\n".join(top_passages)

    # ðŸ”¹ Appliquer reranking aux trois sources de connaissances
    filtered_path = rerank(response_of_KG_list_path)
    filtered_neighbor = rerank(response_of_KG_neighbor)

    # ðŸ”¹ Rerank des phrases start/end
    sent_candidates = [sent_start, sent_end]
    pairs = [(input_text, s) for s in sent_candidates if s]
    ce_scores = cross_encoder.predict(pairs)
    reranked_sentences = [s for s, _ in sorted(zip(sent_candidates, ce_scores), key=lambda x: x[1], reverse=True)]

    # S'assurer qu'on retourne les deux dans le bon ordre (mÃªme si un seul est meilleur)
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
    en gardant seulement les Ã©lÃ©ments les plus similaires Ã  la question
    Ã  lâ€™aide des embeddings SciBERT et de la similaritÃ© cosinus.
    """

    device = model.device

    # === 1. Embedding de la question ===
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    question_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1).cpu().numpy()

    # === 2. Extraction des entitÃ©s ===
    def extract_entities(text):
        entities = []
        matches = re.findall(r"(?:'([^']+)'|(\b\w+::[^\s,]+))", text)
        for match in matches:
            entity = match[0] if match[0] else match[1]
            if entity and "::" in entity:
                entities.append(entity)
        return list(set(entities))

    # === 3. Calcul de similaritÃ© entitÃ©-question ===
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

# CrÃ©er un graphe RDF Ã  partir de votre fichier TXT
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
            
            # Extraire les informations de la prÃ©dicat
            pred_parts = predicate_info.split(' - ')
            if len(pred_parts) != 3:
                continue
                
            subj_type, pred_name, obj_type = pred_parts
            
            # CrÃ©er les URIs
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
    Recherche les phrases pour chaque entitÃ© individuellement
    """
    sentences_entity1 = []
    sentences_entity2 = []
    
    # Nettoyer les noms d'entitÃ©s
    entity1_clean = entity1.strip().lower()
    entity2_clean = entity2.strip().lower()
    
    try:
        # Utiliser pandas pour une lecture plus robuste
        df = pd.read_csv(csv_path)
        
        for index, row in df.iterrows():
            e1 = str(row["Entity1_name"]).strip().lower() if pd.notna(row["Entity1_name"]) else ""
            e2 = str(row["Entity2_name"]).strip().lower() if pd.notna(row["Entity2_name"]) else ""
            sent = str(row.get("Sentence_tokenized", "")).strip() if pd.notna(row.get("Sentence_tokenized")) else ""
            
            # VÃ©rifier si la phrase n'est pas vide
            if not sent or sent == "nan" or sent == "":
                continue
            
            # Recherche pour entity1
            if entity1_clean in e1 or entity1_clean in e2:
                sentences_entity1.append(sent)
            
            # Recherche pour entity2  
            if entity2_clean in e1 or entity2_clean in e2:
                sentences_entity2.append(sent)
        
        # Prendre les premiÃ¨res phrases trouvÃ©es (ou toutes si vous voulez)
        sentence_start = " ".join(sentences_entity1[:3]) if sentences_entity1 else ""
        sentence_end = " ".join(sentences_entity2[:3]) if sentences_entity2 else ""
        
        print(f"Sentences trouvÃ©es pour {entity1}: {len(sentences_entity1)}")
        print(f"Sentences trouvÃ©es pour {entity2}: {len(sentences_entity2)}")
        
    except Exception as e:
        print(f"Erreur lecture CSV : {e}")
        sentence_start, sentence_end = "", ""
    
    return sentence_start, sentence_end

from rdflib import Graph, Namespace, URIRef
import re

EX = Namespace("http://example.org/")

def safe_uri(label):
    """Nettoie les entitÃ©s pour produire une URI valide."""
    clean = label.strip()
    clean = clean.replace(" ", "_")      # remplace espaces
    clean = re.sub(r"[^a-zA-Z0-9_:.-]", "", clean)  # enlÃ¨ve caractÃ¨res interdits
    return EX[clean]

def get_neighbors(graph, entity_name):
    """Retourne les voisins + relations d'une entitÃ© RDF."""
    
    # Enlever les apostrophes si prÃ©sentes
    if entity_name.startswith("'") and entity_name.endswith("'"):
        clean_entity = entity_name[1:-1]  # Enlever les apostrophes
    elif entity_name.startswith('"') and entity_name.endswith('"'):
        clean_entity = entity_name[1:-1]  # Enlever les guillemets
    else:
        clean_entity = entity_name
    
    # Remplacer les espaces par des underscores pour l'URI
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
    
def prompt_path_finding(path_input, chat):
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

        # DÃ©couper en tokens
        tokens = [t.strip() for t in line.split("->")]

        # Extraire les triplets : (entity, relation, entity)
        for i in range(0, len(tokens) - 2, 2):
            e1 = tokens[i]
            rel = tokens[i+1]
            e2 = tokens[i+2]
            triplets.append(f"{e1} -> {rel} -> {e2}")

    return triplets


def prompt_paths(paths_text, question, chat):
    # Extraire les triplets
    triplets = extract_triplets_from_paths(paths_text)

    if not triplets:
        print("âŒ Aucun triplet extrait depuis les paths")
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
    Extrait proprement les triplets rerankÃ©s du output
    """
    triplets = []
    
    # Nettoyer l'input
    reranked_output = reranked_output.strip()
    
    # Diviser par lignes et chercher les triplets
    lines = reranked_output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Chercher les lignes qui contiennent "Reranked Triple"
        if line.startswith('Reranked Triple'):
            # Extraire la partie aprÃ¨s "Reranked TripleX:"
            if ':' in line:
                content = line.split(':', 1)[1].strip()
            else:
                content = line
            
            # Nettoyer le contenu
            content = content.strip()
            
            # IMPORTANT: VÃ©rifier si le contenu contient dÃ©jÃ  un format triple complet
            # Ex: "Cardiomegaly->Inulin â€”â€”> Cardiomegaly->PRKAA1"
            # Ici nous avons un format spÃ©cial qui reprÃ©sente deux triplets joints
            
            if ' â€”â€”> ' in content:
                # C'est notre cas problÃ©matique: deux triplets joints
                # Ex: "Cardiomegaly->Inulin â€”â€”> Cardiomegaly->PRKAA1"
                parts = content.split(' â€”â€”> ')
                
                # VÃ©rifier chaque partie pour voir si c'est un triplet
                for part in parts:
                    part = part.strip()
                    # VÃ©rifier si c'est un triplet valide
                    if '->' in part and 'xxx' not in part.lower():
                        # Nettoyer le triplet
                        part = part.replace(' -> ', '->').replace('-> ', '->').replace(' ->', '->')
                        
                        # Si c'est dÃ©jÃ  un triplet complet, l'ajouter
                        if part.count('->') >= 1:
                            # Dans ce format, chaque partie pourrait Ãªtre un triplet incomplet
                            # Mais on va quand mÃªme l'ajouter pour analyse
                            triplets.append(part)
            elif '->' in content:
                # C'est un triplet simple
                content = content.replace(' -> ', '->').replace('-> ', '->').replace(' ->', '->')
                
                # VÃ©rifier que c'est un triplet valide
                if 'xxx' not in content.lower():
                    triplets.append(content)
    
    # Filtrer les triplets valides
    valid_triplets = []
    for triplet in triplets:
        triplet = triplet.strip()
        if (triplet and 
            '->' in triplet and 
            len(triplet) > 3 and 
            'xxx' not in triplet.lower()):
            valid_triplets.append(triplet)
    
    # Si on n'a pas trouvÃ© de triplets avec le format standard, essayer une extraction plus agressive
    if not valid_triplets:
        for line in lines:
            line = line.strip()
            # Chercher tout ce qui ressemble Ã  "X->Y" ou "X â€”â€”> Y"
            if '->' in line and 'Reranked Triple' in line:
                # Essayer d'extraire le contenu entre ":" et la fin de la ligne
                if ':' in line:
                    content = line.split(':', 1)[1].strip()
                    # Nettoyer et vÃ©rifier
                    content = content.replace(' â€”â€”> ', '->')
                    if '->' in content and 'xxx' not in content.lower():
                        valid_triplets.append(content)
    
    return valid_triplets


def prompt_neighbor(neighbor, chat):
    """Convert reranked raw neighbor triples into natural-language evidence."""
    if isinstance(neighbor, str):
        triplets = extract_reranked_triplets(neighbor)
        if not triplets:
            # LLaDA returns the selected candidates in their original raw format:
            # entity -> relation -> entity (without a "Reranked Triple" label).
            for line in neighbor.splitlines():
                parts = [part.strip() for part in line.strip().split("->")]
                if len(parts) == 3 and all(parts):
                    triplets.append(" -> ".join(parts))

        if not triplets:
            print("No valid neighbor triples found in the reranker output.")
            return "No relevant neighbor-based evidence found."

        neighbor_text = "\n".join(triplets)
        print("Neighbor triples passed to evidence generation:")
        for triplet in triplets:
            print(f"   {triplet}")
    else:
        neighbor_text = str(neighbor)

    template = """
    There are some knowledge graph triples in the format entity->relationship->entity.

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
    prompt = PromptTemplate(template=template, input_variables=["neighbor"])

    try:
        formatted_prompt = prompt.format(neighbor=neighbor_text)
        return chat.invoke(formatted_prompt)
    except Exception as e:
        print(f"Error in prompt_neighbor: {e}")
        return "Error processing neighbor evidence."


def prompt_neighbor11111(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.

    {neighbor}

    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...

    Output:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["neighbor"]
    )

    formatted_prompt = prompt.format(neighbor=neighbor)
    response_of_KG_neighbor = chat.invoke(formatted_prompt)
    return response_of_KG_neighbor




@torch.no_grad()
def llada_generate(model, prompt_ids, attention_mask=None, steps=24,
                   gen_length=96, block_length=32, temperature=0.0,
                   cfg_scale=0.0, remasking="low_confidence", mask_id=126336):
    """Masked denoising sampler for LLaDA with low-confidence remasking."""
    if temperature != 0.0 or cfg_scale != 0.0 or remasking != "low_confidence":
        raise ValueError("This LLaDA sampler supports temperature=0, cfg_scale=0, and low_confidence remasking only")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of generation blocks")

    steps_per_block = steps // num_blocks
    batch_size, prompt_length = prompt_ids.shape
    device = prompt_ids.device
    output = torch.full(
        (batch_size, prompt_length + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    output[:, :prompt_length] = prompt_ids
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, gen_length),
             dtype=attention_mask.dtype, device=device)], dim=1
        )

    for block_index in range(num_blocks):
        block_start = prompt_length + block_index * block_length
        block_end = block_start + block_length
        base, remainder = divmod(block_length, steps_per_block)

        for step_index in range(steps_per_block):
            mask_positions = output[:, block_start:block_end].eq(mask_id)
            remaining = int(mask_positions.sum().item())
            if remaining == 0:
                break
            transfer_count = base + (1 if step_index < remainder else 0)
            transfer_count = min(transfer_count, remaining)
            if transfer_count == 0:
                continue

            logits = model(output, attention_mask=attention_mask).logits
            logits = logits[:, block_start:block_end, :]
            probabilities = torch.softmax(logits.float(), dim=-1)
            predicted = torch.argmax(logits, dim=-1)
            confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            confidence = confidence.masked_fill(~mask_positions, -torch.inf)

            for batch_index in range(batch_size):
                count = min(transfer_count, int(mask_positions[batch_index].sum().item()))
                if count:
                    positions = torch.topk(confidence[batch_index], k=count).indices
                    output[batch_index, block_start + positions] = predicted[batch_index, positions]

    if output[:, prompt_length:].eq(mask_id).any():
        raise RuntimeError("LLaDA denoising ended with unresolved mask tokens")
    return output


_llada_reranker = None


def _load_llada_reranker():
    """Load and cache the LLaDA model/tokenizer once; errors are intentionally fatal."""
    global _llada_reranker
    if _llada_reranker is not None:
        return _llada_reranker

    if not torch.cuda.is_available():
        raise RuntimeError(
            "LLaDA reranking requires a CUDA GPU in this 4-bit configuration. "
            "No fallback reranker is configured."
        )

    model_name = os.environ.get("LLADA_MODEL_NAME", "llada-fine-tuned")
    try:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _llada_reranker = (model, tokenizer)
        print(f"LLaDA loaded once for reranking: {model_name}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load LLaDA ({model_name}). Install compatible transformers, "
            "accelerate, bitsandbytes, and PyTorch CUDA dependencies."
        ) from exc

    return _llada_reranker


def self_knowledge_retrieval_reranking(neighbor_input, question, chatc):
    """Rank numbered graph triples exclusively with LLaDA masked denoising."""
    lines = [line.strip() for line in str(neighbor_input).splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]

    model, tokenizer = _load_llada_reranker()
    numbered_triples = "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, start=1)
    )
    prompt_text = (
        "Rank these knowledge graph triples by relevance to the question. "
        "Return only the numbers of the five most relevant triples, in ranked order, "
        "separated by commas. Do not copy the triples or add explanations.\n"
        f"Question: {question}\nCandidates:\n{numbered_triples}\nRanked candidate numbers:"
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # One generated token per denoising step keeps this short ranking output stable.
    output_ids = llada_generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=32,
        gen_length=32,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    generated = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )

    selected_indices = []
    for value in re.findall(r"(?<![A-Za-z0-9])(?:[1-9]|1\d|20)(?![A-Za-z0-9])", generated):
        index = int(value)
        if index <= len(lines) and index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) == 5:
            break

    if not selected_indices:
        raise RuntimeError(
            "LLaDA completed denoising but returned no valid candidate numbers. "
            "No fallback reranker is configured. Generated text: " + generated[:500]
        )
    return "\n".join(lines[index - 1] for index in selected_indices)


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def is_unable_to_answer(response):
    # Ã€ remplacer par un check basique ou ignorer cette fonction
    return False  # ou un scoring manuel si nÃ©cessaire



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

def final_answer(input_text, response_path_evidence, response_of_KG_neighbor, chat_bio, chat_llama):
    """
    Fonction amÃ©liorÃ©e pour Biomistral fine-tuned avec Alpaca
    OptimisÃ©e pour mieux gÃ©rer les interactions positives
    """


    # Fusion des preuves
    all_evidence = response_of_KG_neighbor + "\n\n" + response_path_evidence

    # -----------------------------  
    # PHASE 1 : Chain-of-Thought - Version amÃ©liorÃ©e
    # -----------------------------
    prompt_cot = f"""
    Below is a medical question and supporting evidence. Think step by step to answer the question.

    Question: {input_text}

    Evidence:
    {all_evidence}

    Instructions:
    1. First, identify the drug/gene entities in the question
    2. Examine what the evidence says about these specific entities
    3. Look for direct mentions, but also consider indirect/implied relationships
    4. If evidence mentions related pathways, mechanisms, or similar entities, consider how they might apply
    5. In biomedical contexts, absence of evidence is not evidence of absence
    6. Conclude with whether an interaction exists based on your analysis

    Step-by-step reasoning:
    """

    result_CoT = chat_llama.invoke(prompt_cot)
    output_CoT = result_CoT

    # -----------------------------  
    # PHASE 2 : Final Answer - Version simplifiÃ©e pour Alpaca
         # -----------------------------  
    prompt_final = f"""
    ### System Prompt
    You are a biomedical expert. Answer the question using the provided evidence and reasoning.
     
    ### Context
    Question: {input_text}

    Medical Evidence:
    {all_evidence}
     
    Reasoning Summary:
    {output_CoT}
     
   ### Examples of Correct Format (DO NOT COPY CONTENT)
    Example 1:
    Question: Does Chemical and Drug Induced Liver Injury interact with Proguanil: A.yes B.no
    Answer: No. Current evidence does not establish a direct interaction between Proguanil and Chemical and Drug Induced Liver Injury. Although isolated cases     of liver injury have been reported with Proguanil, these rare adverse events are not sufficient to demonstrate a consistent or clinically confirmed    interaction. Therefore, the available evidence does not support a positive interaction.

    Example 2:
    Question: Does Frontometaphyseal dysplasia interact with Metamfetamine: A.yes B.no
    Answer: No. There is no documented evidence demonstrating a direct interaction between Frontometaphyseal dysplasia and Metamfetamine. While patients with      this condition may receive various medical treatments, no pharmacodynamic or pharmacokinetic relationship has been established between the disease and         Metamfetamine. Therefore, the available evidence does not support an interaction.

    Example 3:
    Question: Does Amitriptyline interact with Marfan Syndrome: A.yes B.no
    Answer: No. Although Amitriptyline targets several receptors and enzymes, there is no direct evidence linking these targets to a clinically meaningful         interaction with Marfan Syndrome. The available data do not demonstrate that Amitriptyline affects the underlying disease mechanisms or clinical    progression of Marfan Syndrome. Therefore, there is insufficient evidence to conclude that an interaction exists.

    Example 4:
    Question: Does Prazosin interact with Leukemia-Lymphoma Adult T-Cell: A.yes B.no
    Answer: No. Current medical evidence does not demonstrate a direct interaction between Prazosin and Adult T-Cell Leukemia-Lymphoma. Although Prazosin          binds to adrenergic receptors and other molecular targets, there is no confirmed evidence that these interactions influence the pathogenesis or treatment      of Adult T-Cell Leukemia-Lymphoma. Therefore, the available evidence does not support a clinically relevant interaction. 
    
    ### Important Instructions

    1. Your response MUST begin with exactly one of the following:
       - "No. "
       - "Yes. "
    2. After the Yes/No, provide only 1-2 concise sentences explaining the medical reasoning.
    3. Specifically, do NOT produce outputs such as:
       - "Neighbor-based Evidence 1: ..."
       - "Neighbor-based Evidence 2: ..."
       - Any numbered or labeled evidence list.
    4. Base the answer strictly on the provided evidence and reasoning.
    5. In biomedical contexts, you may infer biologically plausible mechanisms even if they are not explicitly stated in the evidence.
    6. Do NOT repeat, restate, quote, or summarize "Medical Evidence" verbatim in the response.

    ### Now Answer This Question
    Question: {input_text}

    Answer:
    """

    final_result = chat_bio.invoke(prompt_final)
    
    return final_result
    





def final_prompt(input_text, response_path_evidence, response_of_KG_neighbor, chat_llama):
    """
    Fonction amÃ©liorÃ©e pour Biomistral fine-tuned avec Alpaca
    OptimisÃ©e pour mieux gÃ©rer les interactions positives
    """


    # Fusion des preuves
    all_evidence = response_of_KG_neighbor + "\n\n" + response_path_evidence

    # -----------------------------  
    # PHASE 1 : Chain-of-Thought - Version amÃ©liorÃ©e
    # -----------------------------
    prompt_cot = f"""
    Below is a medical question and supporting evidence. Think step by step to answer the question.

    Question: {input_text}

    Evidence:
    {all_evidence}

    Instructions:
    1. First, identify the drug/gene entities in the question
    2. Examine what the evidence says about these specific entities
    3. Look for direct mentions, but also consider indirect/implied relationships
    4. If evidence mentions related pathways, mechanisms, or similar entities, consider how they might apply
    5. In biomedical contexts, absence of evidence is not evidence of absence
    6. Conclude with whether an interaction exists based on your analysis

    Step-by-step reasoning:
    """

    result_CoT = chat_llama.invoke(prompt_cot)
    output_CoT = result_CoT

    # -----------------------------  
    # PHASE 2 : Final Answer - Version simplifiÃ©e pour Alpaca
         # -----------------------------  
    prompt_final = f"""
    ### System Prompt
    You are a biomedical expert. Answer the question using the provided evidence and reasoning.
     
    ### Context
    Question: {input_text}

    Medical Evidence:
    {all_evidence}
     
    Reasoning Summary:
    {output_CoT}
     
    ### Examples of Correct Format (DO NOT COPY CONTENT)
    Example 1:
    Question: Does Chemical and Drug Induced Liver Injury interact with Proguanil: A.yes B.no
    Answer: No. Current evidence does not establish a direct interaction between Proguanil and Chemical and Drug Induced Liver Injury. Although isolated cases     of liver injury have been reported with Proguanil, these rare adverse events are not sufficient to demonstrate a consistent or clinically confirmed    interaction. Therefore, the available evidence does not support a positive interaction.

    Example 2:
    Question: Does Frontometaphyseal dysplasia interact with Metamfetamine: A.yes B.no
    Answer: No. There is no documented evidence demonstrating a direct interaction between Frontometaphyseal dysplasia and Metamfetamine. While patients with      this condition may receive various medical treatments, no pharmacodynamic or pharmacokinetic relationship has been established between the disease and         Metamfetamine. Therefore, the available evidence does not support an interaction.

    Example 3:
    Question: Does Amitriptyline interact with Marfan Syndrome: A.yes B.no
    Answer: No. Although Amitriptyline targets several receptors and enzymes, there is no direct evidence linking these targets to a clinically meaningful         interaction with Marfan Syndrome. The available data do not demonstrate that Amitriptyline affects the underlying disease mechanisms or clinical    progression of Marfan Syndrome. Therefore, there is insufficient evidence to conclude that an interaction exists.

    Example 4:
    Question: Does Prazosin interact with Leukemia-Lymphoma Adult T-Cell: A.yes B.no
    Answer: No. Current medical evidence does not demonstrate a direct interaction between Prazosin and Adult T-Cell Leukemia-Lymphoma. Although Prazosin          binds to adrenergic receptors and other molecular targets, there is no confirmed evidence that these interactions influence the pathogenesis or treatment      of Adult T-Cell Leukemia-Lymphoma. Therefore, the available evidence does not support a clinically relevant interaction. 
    
    ### Important Instructions

    1. Your response MUST begin with exactly one of the following:
       - "No. "
       - "Yes. "
    2. After the Yes/No, provide only 1-2 concise sentences explaining the medical reasoning.
    3. Specifically, do NOT produce outputs such as:
       - "Neighbor-based Evidence 1: ..."
       - "Neighbor-based Evidence 2: ..."
       - Any numbered or labeled evidence list.
    4. Base the answer strictly on the provided evidence and reasoning.
    5. In biomedical contexts, you may infer biologically plausible mechanisms even if they are not explicitly stated in the evidence.
    6. Do NOT repeat, restate, quote, or summarize "Medical Evidence" verbatim in the response.

    ### Now Answer This Question
    Question: {input_text}

    Answer:
    """

    
    
    return prompt_final
    


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
    Explore le graphe pour trouver des entitÃ©s similaires aux entitÃ©s cibles
    """
    MY = Namespace("http://mybiogenont.org/ontology#")
    
    print(f"\nðŸ” EXPLORATION DU GRAPHE POUR LES ENTITÃ‰S: {target_entities}")
    
    # Liste de tous les sujets et objets du graphe
    all_entities = set()
    
    # Collecter toutes les entitÃ©s du graphe
    for triple in rdf_graph:
        subj = str(triple[0])
        obj = str(triple[2])
        
        # Nettoyer les URIs - CORRECTION ICI
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
    
    print(f"ðŸ“Š Graphe contient {len(all_entities)} entitÃ©s uniques")
    
    # Chercher des entitÃ©s similaires
    similar_entities = {}
    
    for target in target_entities:
        target_lower = target.lower()
        similar_entities[target] = []
        
        for entity in all_entities:
            entity_lower = entity.lower()
            
            # CritÃ¨res de similaritÃ©
            if (target_lower in entity_lower or 
                entity_lower in target_lower or
                any(word in entity_lower for word in target_lower.split()) or
                any(word in target_lower for word in entity_lower.split())):
                
                # VÃ©rifier combien de triplets cette entitÃ© a
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
                    print(f"âš ï¸ Erreur comptage pour '{entity}': {e}")
                    similar_entities[target].append({
                        'entity': entity,
                        'triplet_count': 0,
                        'similarity_score': calculate_similarity(target, entity)
                    })
        
        # Trier par similaritÃ© et nombre de triplets
        similar_entities[target].sort(key=lambda x: (-x['similarity_score'], -x['triplet_count']))
    
    return similar_entities


def find_shortest_path_in_graph(rdf_graph, entity1, entity2, max_hops=3):
    """
    Trouve le plus court chemin entre deux entitÃ©s dans le graphe RDF
    """
    print(f"ðŸ” Recherche de chemins entre: '{entity1}' et '{entity2}'")
    
    # Nettoyer les entitÃ©s pour les URIs
    entity1_clean = entity1.replace(" ", "_")
    entity2_clean = entity2.replace(" ", "_")
    
    # RequÃªte SPARQL corrigÃ©e
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
            # Chemins avec 1 intermÃ©diaire (2 sauts)
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
            # Chemins avec 2 intermÃ©diaires (3 sauts)
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
        print(f"ðŸ” {len(results)} rÃ©sultat(s) SPARQL trouvÃ©(s)")
        
        for row in results:
            path_length = int(row.pathLength)
            
            # Fonction de nettoyage
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
            
        # Trier par longueur de chemin
        paths.sort(key=lambda x: x['length'])
        
    except Exception as e:
        print(f"âŒ Erreur SPARQL: {e}")
        import traceback
        traceback.print_exc()
    
    return paths

def calculate_similarity(str1, str2):
    """Calcule une similaritÃ© simple entre deux chaÃ®nes"""
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    # Score basÃ© sur l'inclusion
    if str1_lower == str2_lower:
        return 1.0
    elif str1_lower in str2_lower or str2_lower in str1_lower:
        return 0.8
    else:
        # Compter les mots communs
        words1 = set(str1_lower.split())
        words2 = set(str2_lower.split())
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))

def find_best_entity_matches(rdf_graph, entity1, entity2):
    """
    Trouve les meilleures correspondances pour les entitÃ©s dans le graphe
    """
    similar_entities = explore_similar_entities(rdf_graph, [entity1, entity2])
    
    best_matches = {}
    
    for target, matches in similar_entities.items():
        print(f"\nðŸŽ¯ MEILLEURES CORRESPONDANCES POUR '{target}':")
        
        if matches:
            for i, match in enumerate(matches[:5]):  # Top 5
                print(f"   {i+1}. '{match['entity']}' (score: {match['similarity_score']:.2f}, triplets: {match['triplet_count']})")
            
            best_matches[target] = matches[0]['entity']
        else:
            print(f"   âŒ Aucune correspondance trouvÃ©e pour '{target}'")
            best_matches[target] = None
    
    return best_matches


def find_entity_neighbors(rdf_graph, entity_name, limit=10):
    """
    Trouve tous les voisins d'une entitÃ© - VERSION CORRIGÃ‰E
    """
    entity_uri = entity_name.replace(" ", "_")
    
    # RequÃªte SPARQL corrigÃ©e
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
            # Nettoyer les URIs
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
        print(f"âŒ Erreur recherche voisins: {e}")
    
    return neighbors

import re

def clean_uri(value):
    """
    Nettoie une chaÃ®ne pour Ãªtre utilisable comme URI :
    - remplace les espaces par '_'
    - supprime les caractÃ¨res interdits
    - remplace flÃ¨ches et symboles par '_'
    """
    if value is None:
        return "UNKNOWN"

    value = str(value).strip()

    # Remplacer les flÃ¨ches et caractÃ¨res spÃ©ciaux
    value = value.replace("->", "_")
    value = value.replace("=>", "_")

    # Remplacer les espaces par _
    value = value.replace(" ", "_")

    # Remplacer tout ce qui n'est pas alphanumÃ©rique ni _ par _
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)

    # Ã‰viter les URI vides
    if value == "":
        return "EMPTY"

    return value








if __name__ == "__main__":
    print("ðŸš€ Starting main pipeline")

    # Load/download LLaDA once at startup; the reranking function reuses this cached instance.
    _load_llada_reranker()

    chat_bio = load_llm()
    chat_llama = load_cot()

     
    # Charger le graphe RDF depuis le fichier TXT
    rdf_graph = load_txt_to_rdf("SUBDRKG_final.txt")


    with open('SUBDRKG_embeddings.pkl', 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open('fdata_embeddings.pkl', 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    output_path = 'fdata_neg_01_llada_sub.jsonl'
    input_path = "Fdataset-lada-neg.txt"

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

            print(f"\nðŸ”¹ Processing question: {question}")

            # --- Extraction des entitÃ©s ---
            question_kg = []
            entity1 = entity_list[0].split(',')[0].split('1.')[1].strip()
            entity2 = entity_list[0].split(',')[1].split('2.')[1].strip()
            question_kg.append([entity1, entity2])

            #match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
   

            keyword_to_index = {kw: i for i, kw in enumerate(keyword_embeddings["keyword"])}

            # 2) Normalize entity embeddings once
            entity_emb_matrix = np.array(entity_embeddings_emb)
            entity_emb_norm = entity_emb_matrix / np.linalg.norm(entity_emb_matrix, axis=1, keepdims=True)

            # ---------------------------------------------------------

            match_kg = []

            for kg_entity_pair in question_kg:
                pair_matches = []

                for kg_entity in kg_entity_pair:
                    # Retrieve embedding index in O(1)
                    kw_index = keyword_to_index[kg_entity]

                    # Normalized embedding of the KG entity
                    kg_emb = keyword_embeddings["embeddings"][kw_index]
                    kg_emb = kg_emb / np.linalg.norm(kg_emb)

                    # FAST cosine similarity: dot product (already normalized)
                    cos_sim = entity_emb_norm @ kg_emb

                    # best match
                    best_idx = np.argmax(cos_sim)
                    best_match = entity_embeddings["entities"][best_idx]

                    # Avoid duplicates in the pair
                    while best_match in pair_matches:
                        cos_sim[best_idx] = -1
                        best_idx = np.argmax(cos_sim)
                        best_match = entity_embeddings["entities"][best_idx]

                    pair_matches.append(best_match)

                match_kg.append(pair_matches)
              
            first_item = match_kg[0][0]

            # --- Voisins du graphe ---
            EX = Namespace("http://example.org/")

            rdf_graph = Graph()

            with open("SUBDRKG_final.txt", "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 3:
                        continue

                    # Stocker les entitÃ©s ORIGINALES sans modification
                    s = row[0]  # Garder "salmeterol" tel quel
                    p = row[1]
                    o = row[2]

                    # Utiliser les noms originaux pour crÃ©er les URIs
                    subj = URIRef(EX[clean_uri(s)])
                    pred = URIRef(EX[clean_uri(p)])
                    obj  = URIRef(EX[clean_uri(o)])

                    rdf_graph.add((subj, pred, obj))

            print("RDF loaded:", len(rdf_graph), "triples")

            # Appel de la fonction - utiliser first_item directement
            entity = first_item  # Utiliser 'salmeterol' directement
            neighbors = get_neighbors(rdf_graph, entity)
            print("\nðŸŸ¦ Neighbors returned:")
            print(neighbors)


            """  
            # --- RECHERCHE DE CHEMINS ENTRE LES ENTITÃ‰S ---
            result_path = {}

            # VÃ©rifier que match_kg contient au moins UNE paire avec 2 entitÃ©s
            if len(match_kg) > 0 and len(match_kg[0]) >= 2:
    
                ent1, ent2 = match_kg[0][0], match_kg[0][1]

                print(f"\nðŸ” Recherche de chemins entre: {ent1} et {ent2}")

                # Lancer la recherche
                paths = find_shortest_path(ent1, ent2, max_hops=3)
    
                if paths:
                    print("âœ… Chemins trouvÃ©s:")
                    for i, path in enumerate(paths):
                        print(f"   Chemin {i+1} (longueur {path['length']}): {format_path_for_display(path)}")

                    # Garder le chemin le plus court
                    shortest_path = paths[0]
                    result_path = {
                        'entities': shortest_path['entities'],
                        'relations': shortest_path['relations']
                    }

                else:
                    print("âŒ Aucun chemin trouvÃ© entre les entitÃ©s")
                    result_path = {}

            else:
                print("âš ï¸ Pas assez d'entitÃ©s pour chercher un chemin")
                result_path = {}
            """


            entity1 = match_kg[0][0]   # salmeterol
            entity2 = match_kg[0][1]   # adrb2
            # --- APPROCHE SIMPLIFIÃ‰E ---
            #entity1 = entity1
            #entity2 = entity2

            print(f"\nðŸŽ¯ APPROCHE SIMPLIFIÃ‰E POUR: '{entity1}' ET '{entity2}'")


            print(f"\nðŸ” ENTITÃ‰S Ã€ UTILISER:")
            print(f"   '{entity1}' â†’ '{entity1}'")
            print(f"   '{entity2}' â†’ '{entity2}'")

            # Chercher directement un chemin
            print(f"\nðŸ” RECHERCHE DIRECTE DE CHEMINS...")
            paths = find_shortest_path_in_graph(rdf_graph, entity1, entity2, max_hops=3)
            if paths:
                print(f"\nâœ… {len(paths)} CHEMIN(S) TROUVÃ‰(S)")
                for i, path in enumerate(paths):
                    display = f"Chemin {i+1} (longueur {path['length']}): "
                    entities = path['entities']
                    relations = path['relations']
        
                    for j in range(len(relations)):
                        display += f"{entities[j]} --[{relations[j]}]--> "
                    display += entities[-1]
                    print(f"   {display}")
    
                result_path = paths[0]
            else:
                print("âŒ AUCUN CHEMIN TROUVÃ‰")
                result_path = None                    

           
            # --- Knowledge Graph path-based ---
            if len(match_kg) >= 2:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    result_new_path = ["->".join(p) for p in result_path]
                    path = "\n".join(result_new_path)
                    path_sampled = self_knowledge_retrieval_reranking(path, input_text[0], chat_llama)
                    response_of_KG_list_path = prompt_path_finding(path_sampled, chat_llama)
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path_sampled, chat_llama)
            else:
                response_of_KG_list_path = '{}'

            """
            for i, p in enumerate(paths, start=1):
                entities = p["entities"]
                relations = p["relations"]
                length = p["length"]

                # Construire la ligne du chemin
                line = f"Chemin {i} (longueur {length}): "

                for j in range(len(relations)):
                    line += f"{entities[j]} --[{relations[j]}]--> "
                line += entities[-1]
             """

            # Construire une liste de lignes
            graph_lines = []

            for i, p in enumerate(paths, start=1):
                entities = p["entities"]
                relations = p["relations"]
                length = p["length"]

                line = f"Chemin {i} (longueur {length}): "
                for j in range(len(relations)):
                    line += f"{entities[j]} -> {relations[j]} -> "
                line += entities[-1]

                graph_lines.append(line)

            # Transformer en un seul texte pour la fonction
            path = "\n".join(graph_lines)
            

            response_of_KG_list_path = self_knowledge_retrieval_reranking(path, input_text[0], chat_llama)
            response_path_evidence = prompt_paths(response_of_KG_list_path, input_text[0], chat_llama)
            
            # --- Knowledge Graph neighbor-based ---
            #neighbor_new_list = ["->".join(n) for n in neighbors]
            neighbor_new_list = []
            for neighbor_dict in neighbors:
                entity = neighbor_dict['entity']
                # Trouver la clÃ© qui n'est pas 'entity'
                relation_key = [k for k in neighbor_dict.keys() if k != 'entity'][0]
                neighbor_entity = neighbor_dict[relation_key]
                neighbor_new_list.append(f"{entity}->{relation_key}->{neighbor_entity}")   
            #pdb.set_trace()
            #print("\nðŸŸ¦ Neighbor list:")
            #for item in neighbor_new_list:
            #    print("   " + item)
            neighbor_input = "\n".join(neighbor_new_list[:5]) if len(neighbor_new_list) > 5 else "\n".join(neighbor_new_list)
            neighbor_input_sampled = self_knowledge_retrieval_reranking(neighbor_input, input_text[0], chat_llama)
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled, chat_llama)
            #pdb.set_trace()
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled, chat_llama)
                

            # --- RÃ©ponse finale ---
            output_all = final_answer(input_text[0], response_path_evidence, response_of_KG_neighbor, chat_bio, chat_llama)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text[0], response_path_evidence, response_of_KG_neighbor, chat_bio, chat_llama)
            #pdb.set_trace()
            match = re.search(r'(the answer is ([A-D])\.)', output_all.lower())
            if not match:
                match = re.search(r'(\n    ([A-D])\.)', output_all)
            predicted_answer = match.group(2) if match else "Not found"
            

            reasoning = final_prompt(input_text[0], response_path_evidence, response_of_KG_neighbor, chat_llama)

            ret_parsed = {
                "input": input_text[0],
                "answer": predicted_answer,
                "output": output_all,
                "reasoning": reasoning,
                "result_path": response_path_evidence,
                "response_of_KG_neighbor": response_of_KG_neighbor
            }

            # --- âœ… Sauvegarde immÃ©diate du rÃ©sultat ---
            with open(output_path, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(ret_parsed, ensure_ascii=False) + "\n")

            print(f"âœ… RÃ©sultat sauvegardÃ© pour le couple : {drug} - {go_term}")

        except Exception as e:
            print(f"âš ï¸ Erreur lors du traitement du couple {line.strip()} : {e}")
            continue



print("ðŸš€ Main script reached the end")
              

