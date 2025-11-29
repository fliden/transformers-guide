#!/usr/bin/env python3
"""
Text Classification WITH Pipelines
===================================

This script shows the same sentiment classification task using the high-level
pipeline API. Compare this with 02-manual/text_classification.py to see what
the pipeline abstracts away.

The pipeline handles:
- Loading the tokenizer and model
- Tokenizing your input
- Running inference with no_grad
- Applying softmax to get probabilities
- Formatting the output nicely

All in ~5 lines of code.
"""

from transformers import pipeline

# Create the sentiment analysis pipeline
# Explicitly specify the model to avoid warnings about default selection
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Single prediction
text = "An absolutely brilliant film with outstanding performances and a gripping storyline."
result = classifier(text)

print(f"Input: {text}")
print(f"Result: {result}")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch prediction (multiple texts at once)
texts = [
    "I absolutely loved this movie! Best film of the year.",
    "This was a complete waste of time. Terrible acting.",
    "The weather is nice today.",
]

results = classifier(texts)

print("\nBatch predictions:")
for text, result in zip(texts, results):
    print(f"\nText: {text}")
    print(f"  → {result['label']} ({result['score']:.2%})")


# =============================================================================
# That's it! 
#
# But now you know what's happening under the hood:
#   1. Text is tokenized into input_ids and attention_mask
#   2. Model produces logits (raw scores)
#   3. Softmax converts logits to probabilities
#   4. Argmax picks the winning class
#   5. Result is formatted as {'label': '...', 'score': ...}
#
# See 02-manual/text_classification.py for the full breakdown.
# =============================================================================