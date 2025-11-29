#!/usr/bin/env python3
"""
Fill-Mask WITH Pipelines
========================

This script demonstrates the fill-mask task, which is the actual pretraining
objective for BERT-style models. Unlike classification (a downstream task),
this shows what the model was trained to do: predict masked words.

Model: distilbert-base-uncased (or bert-base-uncased)
Task: Predict the [MASK] token in a sentence

This is how BERT learns language:
- Take a sentence
- Randomly mask 15% of words
- Train the model to predict them

The result is a model that "understands" language well enough to be
fine-tuned for downstream tasks (classification, Q&A, etc.)
"""

from transformers import pipeline

# Create the fill-mask pipeline
# Using DistilBERT (smaller, faster, 95% of BERT's performance)
fill_mask = pipeline(
    "fill-mask",
    model="distilbert-base-uncased"
)

# Predict the masked word
prompt = "The capital of France is [MASK]."
results = fill_mask(prompt)

print(f"Prompt: {prompt}")
print(f"\nTop predictions:")
for result in results:
    print(f"  {result['token_str']:12} (score: {result['score']:.4f})")
    print(f"    → \"{result['sequence']}\"")


# =============================================================================
# HOW BERT WAS TRAINED (Masked Language Modeling)
# =============================================================================
#
# GPT-2 training objective:
#   "The cat sat on the" → predict "mat"
#   (Always predict the NEXT token)
#
# BERT training objective:
#   "The cat [MASK] on the mat" → predict "sat"
#   (Predict MASKED tokens anywhere in the sentence)
#
# Why mask instead of next-token?
#   - BERT can see context from BOTH directions (bidirectional)
#   - GPT-2 can only see context from the LEFT (unidirectional)
#   - Bidirectional context is better for understanding, not generation
#
# Trade-off:
#   - BERT: Better at understanding → classification, Q&A, NER
#   - GPT-2: Better at generating → text completion, chat
#
# =============================================================================


# =============================================================================
# MORE EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("More Fill-Mask Examples")
print("="*60)

examples = [
    "Python is a [MASK] programming language.",
    "The weather today is [MASK].",
    "I went to the [MASK] to buy some groceries.",
    "She is a talented [MASK] who plays in an orchestra.",
    "The [MASK] barked loudly at the mailman.",
]

for prompt in examples:
    results = fill_mask(prompt)
    top = results[0]  # Just show top prediction
    print(f"\n\"{prompt}\"")
    print(f"  → {top['token_str']} ({top['score']:.1%})")


# =============================================================================
# PROBING MODEL KNOWLEDGE
# =============================================================================
#
# Fill-mask is useful for exploring what the model "knows":
#
# - Factual knowledge: "The capital of France is [MASK]." → Paris
# - Word associations: "The doctor prescribed [MASK]." → medication, medicine
# - Grammar: "She [MASK] to the store yesterday." → went, walked
# - Bias detection: "The nurse was a [MASK]." → woman? man?
#
# This is how researchers probe language models for biases and knowledge.
#
# =============================================================================

print("\n" + "="*60)
print("Probing Model Knowledge")
print("="*60)

probes = [
    "The Great Wall is located in [MASK].",
    "Water freezes at [MASK] degrees Celsius.",
    "The CEO announced the company would [MASK] more employees.",
]

for prompt in probes:
    results = fill_mask(prompt)
    print(f"\n\"{prompt}\"")
    print("  Top 3 predictions:")
    for r in results[:3]:
        print(f"    {r['token_str']:12} ({r['score']:.1%})")


# =============================================================================
# BERT VS DISTILBERT
# =============================================================================
#
# | Model                    | Parameters | Size    | Speed    |
# |--------------------------|------------|---------|----------|
# | bert-base-uncased        | 110M       | ~420 MB | Baseline |
# | distilbert-base-uncased  | 66M        | ~256 MB | ~60% faster |
#
# DistilBERT retains 97% of BERT's language understanding while being
# 40% smaller and 60% faster. It was trained using "knowledge distillation"
# where the smaller model learns to mimic the larger model's behavior.
#
# For most use cases, DistilBERT is the better choice unless you need
# that extra 3% of performance.
#
# =============================================================================


# =============================================================================
# MULTIPLE MASKS
# =============================================================================

print("\n" + "="*60)
print("Multiple Masks")
print("="*60)

# Note: Most models fill one mask at a time
# If you have multiple [MASK], it predicts each independently
prompt = "I love to [MASK] and [MASK]."
results = fill_mask(prompt)

print(f"\nPrompt: \"{prompt}\"")
print("Note: With multiple [MASK], predictions are for the first one:")
for r in results[:3]:
    print(f"  {r['token_str']:12} ({r['score']:.1%})")


# =============================================================================
# That's it!
#
# Fill-mask shows the PRETRAINING objective of encoder models.
# All those classification, Q&A, and NER models? They start with a
# model trained on fill-mask, then fine-tune for the specific task.
#
# See 02-manual/fill_mask.py for the full breakdown of what's happening.
# =============================================================================