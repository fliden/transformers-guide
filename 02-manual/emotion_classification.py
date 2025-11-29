#!/usr/bin/env python3
"""
Emotion Classification Without Pipelines
=========================================

This script demonstrates multi-class emotion classification using the manual
approach. Unlike binary sentiment (2 classes), emotion classification predicts
one of 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.

Model: j-hartmann/emotion-english-distilroberta-base
Architecture: DistilRoBERTa with a 7-class classification head

This example shows how the process differs when you have more than 2 classes.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# STEP 1: LOAD MODEL AND TOKENIZER
# =============================================================================
#
# We use the Auto classes here instead of model-specific classes.
# AutoTokenizer and AutoModelForSequenceClassification automatically detect
# the correct class based on the model's config.
#
# This is more flexible than using RobertaTokenizer directly.

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")

# The model knows its own labels - let's see them
print(f"Number of classes: {model.config.num_labels}")
print(f"Labels: {model.config.id2label}")


# =============================================================================
# STEP 2: PREPARE INPUT
# =============================================================================

text = "I can't believe they surprised me with a birthday party!"

print(f"\nInput text: {text}")

inputs = tokenizer(text, return_tensors="pt")

print(f"\n--- Tokenization ---")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[:10]}...")


# =============================================================================
# STEP 3: RUN INFERENCE
# =============================================================================

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

print(f"\n--- Raw Logits ---")
print(f"Logits shape: {logits.shape}")  # [batch_size, num_classes] = [1, 7]
print(f"Logits: {logits}")


# =============================================================================
# STEP 4: CONVERT TO PROBABILITIES
# =============================================================================
#
# With 7 classes, softmax still works the same way:
# - All 7 probabilities will be between 0 and 1
# - All 7 probabilities will sum to 1.0

probs = torch.softmax(logits, dim=-1)

print(f"\n--- Probabilities (after softmax) ---")
print(f"Sum of probabilities: {probs.sum():.4f}")  # Should be ~1.0

# Display all emotions with their probabilities
print(f"\nEmotion scores for: \"{text}\"")
for idx, prob in enumerate(probs[0]):
    label = model.config.id2label[idx]
    bar = "█" * int(prob.item() * 20)
    print(f"  {label:8} {prob.item():.2%} {bar}")


# =============================================================================
# STEP 5: GET PREDICTED CLASS
# =============================================================================
#
# Same as binary classification - argmax finds the highest probability

predicted_idx = torch.argmax(probs, dim=-1).item()
predicted_emotion = model.config.id2label[predicted_idx]
confidence = probs[0][predicted_idx].item()

print(f"\n--- Prediction ---")
print(f"Predicted emotion: {predicted_emotion}")
print(f"Confidence: {confidence:.2%}")


# =============================================================================
# MULTI-CLASS VS BINARY CLASSIFICATION
# =============================================================================
#
# The process is identical! The only differences are:
#
# 1. Logits shape: [batch, 2] for binary vs [batch, 7] for 7-class
# 2. Softmax output: 2 probabilities vs 7 probabilities
# 3. Labels: model.config.id2label maps indices to emotion names
#
# Everything else - tokenization, inference, softmax, argmax - stays the same.
#
# =============================================================================


# =============================================================================
# MORE EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("More Examples")
print("="*60)

examples = [
    "I'm so happy I could dance!",
    "This is absolutely disgusting.",
    "I'm terrified of what might happen.",
    "I can't believe they lied to me!",
    "The meeting is at 3pm.",
    "I miss her so much.",
    "Wait, what just happened?!",
]

for text in examples:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    emotion = model.config.id2label[pred_idx]
    confidence = probs[0][pred_idx].item()
    
    print(f"\n\"{text}\"")
    print(f"  → {emotion} ({confidence:.1%})")


# =============================================================================
# TOP-K PREDICTIONS
# =============================================================================
#
# Sometimes you want the top 3 emotions, not just the winner.
# Use torch.topk() for this:

print("\n" + "="*60)
print("Top-3 Emotions Example")
print("="*60)

text = "I can't believe they surprised me with a birthday party!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.softmax(outputs.logits, dim=-1)
top_k = torch.topk(probs[0], k=3)

print(f"\nText: \"{text}\"")
print(f"\nTop 3 emotions:")
for score, idx in zip(top_k.values, top_k.indices):
    emotion = model.config.id2label[idx.item()]
    print(f"  {emotion}: {score.item():.2%}")