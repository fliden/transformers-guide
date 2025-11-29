#!/usr/bin/env python3
"""
Fill-Mask Without Pipelines
===========================

This script demonstrates the fill-mask task using the manual approach.
Fill-mask is the pretraining objective for BERT-style models — this is
how they learn to "understand" language.

Model: distilbert-base-uncased
Task: Predict the [MASK] token in a sentence

This example shows:
- How masked language modeling works
- The difference between BERT (bidirectional) and GPT-2 (unidirectional)
- How to find the mask token position and extract predictions
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# =============================================================================
# STEP 1: LOAD MODEL AND TOKENIZER
# =============================================================================
#
# AutoModelForMaskedLM: A model with a "masked language modeling head"
# - The head predicts which token should go in the [MASK] position
# - Output shape: [batch, sequence_length, vocab_size]
# - At each position, we get scores for all ~30,000 tokens in vocabulary
#
# Note: This is the BASE model, not fine-tuned for any downstream task.
# Classification models (like the sentiment one) ADD a classification head
# on top of this base.

MODEL_NAME = "distilbert-base-uncased"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")
print(f"Parameters: {model.num_parameters():,}")
print(f"Vocabulary size: {tokenizer.vocab_size:,}")


# =============================================================================
# STEP 2: UNDERSTAND THE MASK TOKEN
# =============================================================================
#
# BERT-style models use a special [MASK] token.
# During pretraining, random words are replaced with [MASK], and the model
# learns to predict what word should be there.

print(f"\n--- Special Tokens ---")
print(f"Mask token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")
print(f"CLS token: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
print(f"SEP token: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")


# =============================================================================
# STEP 3: TOKENIZE INPUT WITH MASK
# =============================================================================

text = "The capital of France is [MASK]."

print(f"\n--- Input ---")
print(f"Text: {text}")

inputs = tokenizer(text, return_tensors="pt")

print(f"\n--- Tokenization ---")
print(f"Input IDs: {inputs['input_ids']}")

# Let's see the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"Tokens: {tokens}")

# Find where the [MASK] token is
mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
print(f"[MASK] position: {mask_token_index.item()}")


# =============================================================================
# STEP 4: RUN INFERENCE
# =============================================================================
#
# The model outputs logits for EVERY position in the sequence.
# Shape: [batch_size, sequence_length, vocab_size]
#
# For fill-mask, we only care about the logits at the [MASK] position.

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print(f"\n--- Model Output ---")
print(f"Logits shape: {logits.shape}")
print(f"  - Batch size: {logits.shape[0]}")
print(f"  - Sequence length: {logits.shape[1]}")
print(f"  - Vocabulary size: {logits.shape[2]}")


# =============================================================================
# STEP 5: EXTRACT PREDICTIONS FOR THE MASK
# =============================================================================
#
# We need to:
# 1. Get the logits at the [MASK] position
# 2. Apply softmax to get probabilities
# 3. Find the top-k most likely tokens

# Get logits only at the mask position
mask_logits = logits[0, mask_token_index, :].squeeze()  # Shape: [vocab_size]

print(f"\n--- Mask Position Logits ---")
print(f"Shape: {mask_logits.shape}")  # [vocab_size]

# Convert to probabilities
probs = torch.softmax(mask_logits, dim=-1)

# Get top 5 predictions
top_k = torch.topk(probs, k=5)

print(f"\n--- Top 5 Predictions ---")
print(f"Text: \"{text}\"")
print()
for score, token_id in zip(top_k.values, top_k.indices):
    token = tokenizer.decode(token_id)
    # Show the complete sentence with this token
    filled_text = text.replace("[MASK]", token)
    print(f"  {token:12} ({score.item():.2%})")
    print(f"    → \"{filled_text}\"")


# =============================================================================
# HOW BERT LEARNS: BIDIRECTIONAL CONTEXT
# =============================================================================
#
# The key insight of BERT is BIDIRECTIONAL attention.
#
# GPT-2 (unidirectional):
#   "The capital of France is ___"
#   Can only see: "The capital of France is" (left context)
#
# BERT (bidirectional):
#   "The capital of France is [MASK]."
#   Can see: "The capital of France is" AND "." (both sides!)
#
# Why does this matter?
#
# Consider: "The ___ barked at the mailman."
#   - Unidirectional: "The" → could be anything
#   - Bidirectional: "The" + "barked at the mailman" → probably "dog"
#
# Bidirectional context helps the model understand language better,
# which is why BERT-based models excel at understanding tasks.
#
# =============================================================================


print("\n" + "="*60)
print("BIDIRECTIONAL CONTEXT DEMO")
print("="*60)

# The word "bank" has multiple meanings
# Context from BOTH sides helps disambiguate

examples = [
    "I went to the [MASK] to deposit my money.",      # bank (financial)
    "I sat by the river [MASK] to watch the sunset.", # bank (riverbank)
    "The [MASK] approved my loan application.",       # bank (financial)
]

print("\nSame word, different contexts:\n")
for text in examples:
    inputs = tokenizer(text, return_tensors="pt")
    mask_idx = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_logits = outputs.logits[0, mask_idx, :].squeeze()
    probs = torch.softmax(mask_logits, dim=-1)
    top_token_id = torch.argmax(probs)
    top_token = tokenizer.decode(top_token_id)
    
    print(f"\"{text}\"")
    print(f"  → Prediction: {top_token} ({probs[top_token_id].item():.1%})\n")


# =============================================================================
# COMPARING ARCHITECTURES
# =============================================================================
#
# | Aspect              | BERT / DistilBERT        | GPT-2                    |
# |---------------------|--------------------------|--------------------------|
# | Training objective  | Masked Language Model    | Next Token Prediction    |
# | Attention           | Bidirectional            | Unidirectional (causal)  |
# | Sees context from   | Both left AND right      | Only left                |
# | Best for            | Understanding tasks      | Generation tasks         |
# | Examples            | Classification, Q&A, NER | Text completion, chat    |
#
# Both are "language models" but trained differently:
# - BERT: "Fill in the blank" (cloze task)
# - GPT-2: "What comes next?" (completion task)
#
# =============================================================================


# =============================================================================
# MORE EXAMPLES
# =============================================================================

print("="*60)
print("More Examples")
print("="*60)

examples = [
    "She went to the [MASK] to buy some bread.",
    "The [MASK] flew across the blue sky.",
    "Python is a popular [MASK] language.",
    "The doctor prescribed some [MASK] for the patient.",
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt")
    mask_idx = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_logits = outputs.logits[0, mask_idx, :].squeeze()
    probs = torch.softmax(mask_logits, dim=-1)
    
    top_k = torch.topk(probs, k=3)
    
    print(f"\n\"{text}\"")
    print("  Top 3:", end=" ")
    predictions = [tokenizer.decode(idx) for idx in top_k.indices]
    print(", ".join(predictions))


# =============================================================================
# WHY THIS MATTERS
# =============================================================================
#
# Fill-mask isn't just a party trick — it's the foundation of modern NLP:
#
# 1. PRETRAINING: Models learn language by predicting masked words
#    on massive amounts of text (Wikipedia, books, web pages)
#
# 2. FINE-TUNING: The pretrained model is adapted for specific tasks:
#    - Add a classification head → sentiment analysis
#    - Add a Q&A head → question answering
#    - Add a token classification head → named entity recognition
#
# 3. TRANSFER LEARNING: The language "understanding" transfers to new tasks,
#    even with limited task-specific training data
#
# This pretrain-then-finetune paradigm revolutionized NLP!
#
# =============================================================================

print("\n" + "="*60)
print("The Pretrain → Fine-tune Pipeline")
print("="*60)

print("""
┌─────────────────────────────────────────────────────────────┐
│                     PRETRAINING                             │
│                                                             │
│   Massive text corpus (Wikipedia, books, web)               │
│                    ↓                                        │
│   Mask random words: "The [MASK] sat on the [MASK]"         │
│                    ↓                                        │
│   Train to predict: "cat", "mat"                            │
│                    ↓                                        │
│   Result: Model that "understands" language                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     FINE-TUNING                             │
│                                                             │
│   Take pretrained model                                     │
│                    ↓                                        │
│   Add task-specific head (classification, Q&A, etc.)        │
│                    ↓                                        │
│   Train on small labeled dataset                            │
│                    ↓                                        │
│   Result: Model that does your specific task well!          │
└─────────────────────────────────────────────────────────────┘

This is why DistilBERT can do sentiment analysis, Q&A, NER, and more —
they all start from the same fill-mask pretrained model!
""")