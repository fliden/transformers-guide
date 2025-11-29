#!/usr/bin/env python3
"""
Text Classification Without Pipelines
======================================

This script demonstrates how to perform text classification using HuggingFace
Transformers WITHOUT the high-level pipeline API. This approach gives you full
control over the tokenization, inference, and post-processing steps.

Model: DistilBERT fine-tuned on SST-2 (Stanford Sentiment Treebank)
Task: Binary sentiment classification (POSITIVE / NEGATIVE)

Prerequisites:
    pip install transformers torch

For Apple Silicon (M1/M2/M3) Macs:
    The standard torch installation includes MPS (Metal Performance Shaders)
    support automatically. Verify with:
    >>> import torch
    >>> torch.backends.mps.is_available()  # Should return True
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


# =============================================================================
# STEP 1: LOAD THE MODEL AND TOKENIZER
# =============================================================================
#
# Every Transformer model needs two components:
#
# 1. TOKENIZER: Converts text into numbers (token IDs) that the model understands
#    - Splits text into subwords or words
#    - Maps each piece to a unique integer ID
#    - Handles special tokens like [CLS] and [SEP]
#
# 2. MODEL: The neural network that processes token IDs and produces predictions
#    - DistilBERT is a smaller, faster version of BERT (40% smaller, 60% faster)
#    - "ForSequenceClassification" means it has a classification head on top
#    - This specific model was fine-tuned on movie review sentiment data

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")


# =============================================================================
# STEP 2: PREPARE INPUT TEXT
# =============================================================================
#
# The tokenizer converts your text into a format the model can process:
# - input_ids: The token ID for each piece of text
# - attention_mask: Tells the model which tokens are real (1) vs padding (0)
#
# return_tensors="pt" means "return PyTorch tensors" (as opposed to numpy arrays
# or TensorFlow tensors)

text = "An absolutely brilliant film with outstanding performances and a gripping storyline."

print(f"\nInput text: {text}")

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Let's peek at what the tokenizer produced
print(f"\n--- Tokenization Results ---")
print(f"Input IDs shape: {inputs['input_ids'].shape}")  # [batch_size, sequence_length]
print(f"Token count: {inputs['input_ids'].shape[1]}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")

# Optionally: See the actual tokens (useful for debugging)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens


# =============================================================================
# STEP 3: RUN INFERENCE
# =============================================================================
#
# Key concepts:
#
# torch.no_grad(): Tells PyTorch we're NOT training, just making predictions.
#   - Disables gradient computation (not needed for inference)
#   - Reduces memory usage and speeds up computation
#
# model(**inputs): The ** unpacks the dictionary as keyword arguments.
#   - Equivalent to: model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
#
# The model returns a ModelOutput object containing:
#   - logits: Raw prediction scores (NOT probabilities yet!)
#   - Other optional fields depending on the model configuration

print("\n--- Running Inference ---")

with torch.no_grad():
    outputs = model(**inputs)

# Alternative syntax (explicit parameter passing):
# outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])


# =============================================================================
# STEP 4: UNDERSTAND LOGITS
# =============================================================================
#
# WHAT ARE LOGITS?
# ----------------
# Logits are the raw, unnormalized output scores from the model's final layer.
# They are NOT probabilities - they can be any real number (positive or negative).
#
# For binary classification, you get 2 logits:
#   - logits[0][0]: Score for class 0 (NEGATIVE)
#   - logits[0][1]: Score for class 1 (POSITIVE)
#
# Higher logit = model is more confident about that class
#
# WHY LOGITS INSTEAD OF PROBABILITIES?
# ------------------------------------
# Models output logits because:
#   1. They're more numerically stable for training (with cross-entropy loss)
#   2. They preserve more information than squashed probabilities
#   3. You can choose how to convert them based on your needs
#
# Shape: [batch_size, num_classes]
# For single input and binary classification: [1, 2]

logits = outputs.logits

print(f"Logits shape: {logits.shape}")  # [1, 2] for single input, 2 classes
print(f"Raw logits: {logits}")
print(f"  - NEGATIVE score: {logits[0][0]:.4f}")
print(f"  - POSITIVE score: {logits[0][1]:.4f}")


# =============================================================================
# STEP 5: CONVERT LOGITS TO PROBABILITIES
# =============================================================================
#
# SOFTMAX FUNCTION
# ----------------
# Softmax converts logits into probabilities that:
#   1. Are between 0 and 1
#   2. Sum to 1.0 across all classes
#
# Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
#
# The dim=-1 argument means "apply softmax across the last dimension"
# (the class dimension, not the batch dimension)
#
# Example:
#   logits = [-3.5, 4.2]
#   After softmax: [0.0004, 0.9996]  # Now they're probabilities!

probs = torch.softmax(logits, dim=-1)

print(f"\n--- Probabilities (after softmax) ---")
print(f"Probabilities: {probs}")
print(f"  - NEGATIVE: {probs[0][0]:.2%}")
print(f"  - POSITIVE: {probs[0][1]:.2%}")
print(f"Sum of probabilities: {probs.sum():.4f}")  # Should be ~1.0


# =============================================================================
# STEP 6: GET THE PREDICTED CLASS
# =============================================================================
#
# ARGMAX
# ------
# Returns the index of the highest value. For our probabilities:
#   - argmax returns 0 if NEGATIVE probability is higher
#   - argmax returns 1 if POSITIVE probability is higher
#
# .item() converts a single-element tensor to a Python number

predicted_class_idx = torch.argmax(probs, dim=-1).item()

# Map class index to human-readable label
# Note: This label order matches how the model was trained
labels = ["NEGATIVE", "POSITIVE"]
predicted_label = labels[predicted_class_idx]
confidence = probs[0][predicted_class_idx].item()

print(f"\n--- Final Prediction ---")
print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.2%}")


# =============================================================================
# COMPARISON: WHAT THE PIPELINE DOES AUTOMATICALLY
# =============================================================================
#
# The high-level pipeline API wraps all of the above into a single call:
#
#   from transformers import pipeline
#   classifier = pipeline("sentiment-analysis")
#   result = classifier("Your text here")
#   # Returns: [{'label': 'POSITIVE', 'score': 0.9998}]
#
# The pipeline handles:
#   - Loading the tokenizer and model
#   - Tokenizing your input
#   - Running inference with no_grad
#   - Applying softmax to get probabilities
#   - Formatting the output nicely
#
# Use the manual approach when you need:
#   - Access to raw logits (for custom loss functions)
#   - Batch processing with custom logic
#   - Fine-grained control over the inference process
#   - Integration into a larger system
#
# =============================================================================


# =============================================================================
# TRY MORE EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("More Examples")
print("="*60)

test_texts = [
    "I absolutely loved this movie! Best film of the year.",
    "This was a complete waste of time. Terrible acting.",
    "The weather is nice today.",  # Neutral-ish text
]

for test_text in test_texts:
    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    
    print(f"\nText: {test_text}")
    print(f"  → {labels[pred_idx]} ({probs[0][pred_idx]:.2%})")