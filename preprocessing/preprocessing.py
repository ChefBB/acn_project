"""
Preprocessing module
------

Contains functions for preprocessing the data


- correzione OCR (con BERT su macchina di Leo)
- normalizzazione
- lower case
- controllo unicode+rimozione caratteri non supportati
- split in frasi (una per riga)
- gestione parole spezzate
- normalizzazione doppi spazi etc
- normalizzazione apostrofi virgolette accenti ecc
"""

from typing import Optional, List, Sequence, Dict, Any
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)
import os
import re


# import entities
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dataset_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/dataset/dict_all'

with open(f'{dataset_dir}/entity_all_1.txt') as f:
    ENTITIES = set()
    for line in f:
        line = line.strip()
        if not line:
            continue
        first = line.split()[0]
        ENTITIES.add(first)
        
with open(f'{dataset_dir}/entity_all_2.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        first = line.split()[0]
        ENTITIES.add(first)


MODEL = "osiria/blaze-it"
PROB_THRESHOLD = 0.02  # mask tokens with probability lower than this

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def replace_entities_with_placeholders(sentence, entities):
    """
    Replace entities with placeholders safely without regex looping issues.
    Returns processed_sentence and a map placeholder -> entity.
    """
    placeholder_map = {}
    processed_sentence = sentence

    # Sort entities by length descending to replace longer entities first
    entities_sorted = sorted(entities, key=len, reverse=True)
    
    # Track replacements using character indices
    replacements = []
    for i, ent in enumerate(entities_sorted):
        start = 0
        while True:
            idx = processed_sentence.find(ent, start)
            if idx == -1:
                break
            placeholder = f'ENTITY{i}'
            replacements.append((idx, idx+len(ent), placeholder, ent))
            start = idx + len(ent)
    
    # Sort replacements by start index descending so we don't shift earlier indices
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    for start_idx, end_idx, placeholder, ent in replacements:
        processed_sentence = processed_sentence[:start_idx] + placeholder + processed_sentence[end_idx:]
        placeholder_map[placeholder] = ent

    return processed_sentence, placeholder_map



def mask_suspicious_subwords(sentence, prob_threshold=PROB_THRESHOLD, entities=ENTITIES):
    """
    Mask suspicious subwords based on model probability, but treat named entities as single units
    that are not split into subwords and are never masked.

    Args:
        sentence (str): input sentence
        prob_threshold (float): probability threshold for masking
        entities (set of str): full named entities to skip masking

    Returns:
        masked_tokens: list of subwords with [MASK] where suspicious
        masked_word_indices: list of (start_idx, end_idx) for consecutive masked subwords
    """
    processed_sentence, placeholder_map = replace_entities_with_placeholders(sentence, ENTITIES)

    # tokenization
    tokens = tokenizer.tokenize(processed_sentence)
    masked_tokens = []
    masked_word_indices = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        # Skip masking if token is a placeholder
        if tok in placeholder_map:
            masked_tokens.append(tok)
            i += 1
            continue

        # Probability-based masking
        temp_tokens = tokens.copy()
        temp_tokens[i] = tokenizer.mask_token
        input_ids = tokenizer.convert_tokens_to_ids(temp_tokens)
        input_ids = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits[0, i], dim=-1)
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            token_prob = probs[tok_id].item()

        if token_prob < prob_threshold:
            start_idx = len(masked_tokens)
            masked_tokens.append(tokenizer.mask_token)
            j = i + 1
            # Group consecutive subwords starting with '##'
            while j < n and tokens[j].startswith("##"):
                temp_tokens2 = tokens.copy()
                temp_tokens2[j] = tokenizer.mask_token
                input_ids2 = tokenizer.convert_tokens_to_ids(temp_tokens2)
                input_ids2 = torch.tensor([input_ids2]).to(device)
                with torch.no_grad():
                    outputs2 = model(input_ids2)
                    logits2 = outputs2.logits
                    probs2 = torch.softmax(logits2[0, j], dim=-1)
                    tok_id2 = tokenizer.convert_tokens_to_ids(tokens[j])
                    token_prob2 = probs2[tok_id2].item()
                if token_prob2 < prob_threshold:
                    masked_tokens.append(tokenizer.mask_token)
                else:
                    masked_tokens.append(tokens[j])
                j += 1
            end_idx = len(masked_tokens) - 1
            masked_word_indices.append((start_idx, end_idx))
            i = j
        else:
            masked_tokens.append(tok)
            i += 1

    # restore original entities
    masked_tokens = [placeholder_map.get(t, t) for t in masked_tokens]

    return masked_tokens, masked_word_indices


def substitute_masked_tokens(masked_tokens, masked_word_indices):
    """
    Substitute masked words made of multiple subwords as a single unit.
    """
    substituted_tokens = masked_tokens.copy()

    for start, end in masked_word_indices:
        # Prepare input
        input_ids = tokenizer.convert_tokens_to_ids(substituted_tokens)
        input_ids = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # For multiple consecutive masks, predict each mask
        # then combine predictions into a single word
        predicted_tokens = []
        for idx in range(start, end + 1):
            mask_logits = logits[0, idx]
            top_id = torch.topk(mask_logits, 1).indices.item()
            predicted_token = tokenizer.convert_ids_to_tokens(top_id)
            predicted_tokens.append(predicted_token)

        # Replace the masked tokens with the predicted tokens
        substituted_tokens[start:end+1] = predicted_tokens

    # Merge subwords into a string
    substituted_sentence = tokenizer.convert_tokens_to_string(substituted_tokens)
    return substituted_sentence