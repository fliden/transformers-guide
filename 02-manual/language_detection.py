#!/usr/bin/env python3
"""
Language Detection Without Pipelines
=====================================

This script demonstrates language detection using the manual approach
with XLM-RoBERTa, a multilingual transformer model.

Model: papluca/xlm-roberta-base-language-detection
Base: XLM-RoBERTa (multilingual RoBERTa trained on 100 languages)
Task: Classify text into one of 20 supported languages
Accuracy: 99.6% on test set
Size: ~1.1 GB

This example shows:
- Working with multilingual models
- How the same classification process works across languages
- Using model.config.id2label for many classes
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# STEP 1: LOAD MODEL AND TOKENIZER
# =============================================================================
#
# XLM-RoBERTa is a multilingual model:
# - Trained on 2.5TB of CommonCrawl data in 100 languages
# - Uses the same architecture as RoBERTa
# - Learns cross-lingual representations
#
# This fine-tuned version classifies into 20 languages:
# ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh

MODEL_NAME = "papluca/xlm-roberta-base-language-detection"

print("Loading tokenizer and model...")
print("(Note: This model is ~1.1 GB, first download may take a while)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")
print(f"Parameters: {model.num_parameters():,}")

# See all supported languages
print(f"\nSupported languages ({model.config.num_labels}):")
id2label = model.config.id2label
print(", ".join(f"{v} ({k})" for k, v in sorted(id2label.items())))


# =============================================================================
# STEP 2: LANGUAGE DETECTION
# =============================================================================

text = "Bonjour, comment ça va?"

print(f"\n" + "="*60)
print("Language Detection")
print("="*60)
print(f"\nInput text: {text}")

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True)

print(f"\n--- Tokenization ---")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Number of tokens: {inputs['input_ids'].shape[1]}")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print(f"\n--- Logits ---")
print(f"Logits shape: {logits.shape}")  # [batch_size, num_languages] = [1, 20]


# =============================================================================
# STEP 3: CONVERT TO PROBABILITIES
# =============================================================================

probs = torch.softmax(logits, dim=-1)

# Get top prediction
predicted_idx = torch.argmax(probs, dim=-1).item()
predicted_lang = id2label[predicted_idx]
confidence = probs[0][predicted_idx].item()

print(f"\n--- Prediction ---")
print(f"Detected language: {predicted_lang}")
print(f"Confidence: {confidence:.2%}")

# Show top 5 languages
print(f"\n--- Top 5 Languages ---")
top_k = torch.topk(probs[0], k=5)
for score, idx in zip(top_k.values, top_k.indices):
    lang = id2label[idx.item()]
    bar = "█" * int(score.item() * 30)
    print(f"  {lang:4} {score.item():.2%} {bar}")


# =============================================================================
# STEP 4: MULTILINGUAL EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("Multilingual Examples")
print("="*60)

examples = [
    ("Hello, how are you?", "English"),
    ("Hola, ¿cómo estás?", "Spanish"),
    ("Guten Tag, wie geht es Ihnen?", "German"),
    ("こんにちは、お元気ですか？", "Japanese"),
    ("你好，你好吗？", "Chinese"),
    ("Привет, как дела?", "Russian"),
    ("مرحبا، كيف حالك؟", "Arabic"),
]

print()
for text, expected in examples:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    detected = id2label[pred_idx]
    confidence = probs[0][pred_idx].item()
    
    print(f"\"{text}\"")
    print(f"  Expected: {expected}, Detected: {detected} ({confidence:.1%})\n")


# =============================================================================
# WHY USE NEURAL LANGUAGE DETECTION?
# =============================================================================
#
# Traditional methods (like the langid library) use statistical n-gram features.
# Neural models like this one offer advantages:
#
# 1. Better on short text — N-gram methods need more text to be accurate
# 2. Handle similar languages — Portuguese vs Spanish, Norwegian vs Swedish
# 3. Mixed-language text — Can detect when languages are mixed
# 4. Code-switching — Handle sentences that switch between languages
#
# Trade-offs:
# - Larger model size (~1.1 GB vs ~1 MB for langid)
# - Slower inference (though still fast)
# - Limited to trained languages (20 vs langid's 97)
#
# For most applications, the neural approach's accuracy advantage is worth it.
#
# =============================================================================


# =============================================================================
# ABOUT XLM-ROBERTA
# =============================================================================
#
# XLM-RoBERTa ("Cross-lingual Language Model - RoBERTa") is:
#
# 1. A multilingual transformer (like mBERT, but better)
# 2. Trained on 2.5TB of filtered CommonCrawl data
# 3. Covers 100 languages
# 4. Uses the same masked language modeling objective as RoBERTa
#
# Key insight: By training on many languages simultaneously, the model learns
# cross-lingual representations. This means:
#
# - Fine-tuning on English data often works for other languages too
# - The model can transfer knowledge between similar languages
# - It's a good foundation for many multilingual NLP tasks
#
# This specific model (papluca/xlm-roberta-base-language-detection) was
# fine-tuned on a language identification dataset with examples from 20
# languages, achieving 99.6% accuracy.
#
# =============================================================================

print("="*60)
print("Model Architecture")
print("="*60)

print(f"""
XLM-RoBERTa Language Detector:

┌─────────────────────────────────────┐
│  Input: "Bonjour, comment ça va?"   │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │   XLM-RoBERTa Encoder       │    │
│  │   (Multilingual, 100 langs) │    │
│  └─────────────────────────────┘    │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │  Classification Head        │    │
│  │  (Linear: hidden → 20)      │    │
│  └─────────────────────────────┘    │
│              ↓                      │
│  Logits: [0.1, 0.2, ..., 8.9, ...]  │
│              ↓                      │
│  Softmax → Probabilities            │
│              ↓                      │
│  Output: "fr" (French) 99.8%        │
└─────────────────────────────────────┘

Same process as sentiment/emotion classification,
just with 20 language classes instead of 2 or 7.
""")