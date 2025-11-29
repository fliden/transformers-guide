#!/usr/bin/env python3
"""
Language Detection WITH Pipelines
==================================

This script demonstrates language detection using a multilingual model.
This is another practical application of text classification — instead of
sentiment or emotion, we're classifying which language the text is written in.

Model: papluca/xlm-roberta-base-language-detection
Base: XLM-RoBERTa (multilingual RoBERTa trained on 100 languages)
Task: Classify text into one of 20 supported languages
Accuracy: 99.6% on test set
Size: ~1.1 GB

Supported languages (20):
Arabic (ar), Bulgarian (bg), German (de), Greek (el), English (en),
Spanish (es), French (fr), Hindi (hi), Italian (it), Japanese (ja),
Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Swahili (sw),
Thai (th), Turkish (tr), Urdu (ur), Vietnamese (vi), Chinese (zh)
"""

from transformers import pipeline

# Create the language detection pipeline
# Note: This model is ~1.1 GB, so first download may take a while
classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)

# Detect language of a French sentence
text = "Bonjour, comment ça va?"
result = classifier(text)

print(f"Text: {text}")
print(f"Detected language: {result[0]['label']} (confidence: {result[0]['score']:.2%})")


# =============================================================================
# MULTILINGUAL EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("Language Detection Examples")
print("="*60)

examples = [
    ("Hello, how are you?", "English"),
    ("Hola, ¿cómo estás?", "Spanish"),
    ("Guten Tag, wie geht es Ihnen?", "German"),
    ("Ciao, come stai?", "Italian"),
    ("こんにちは、お元気ですか？", "Japanese"),
    ("你好，你好吗？", "Chinese"),
    ("Привет, как дела?", "Russian"),
    ("مرحبا، كيف حالك؟", "Arabic"),
    ("Olá, como você está?", "Portuguese"),
    ("Cześć, jak się masz?", "Polish"),
]

print("\nDetecting languages:\n")
for text, expected in examples:
    result = classifier(text, truncation=True)
    detected = result[0]['label']
    confidence = result[0]['score']
    match = "✓" if detected.lower()[:2] in expected.lower() else "?"
    print(f"{match} \"{text}\"")
    print(f"   Expected: {expected}, Detected: {detected} ({confidence:.1%})\n")


# =============================================================================
# ABOUT XLM-ROBERTA
# =============================================================================
#
# XLM-RoBERTa is a multilingual version of RoBERTa:
#
# - Trained on 2.5TB of CommonCrawl data in 100 languages
# - Can be fine-tuned for tasks in any of those languages
# - Learns cross-lingual representations (understanding transfers between languages)
#
# This specific model (papluca/xlm-roberta-base-language-detection) was
# fine-tuned on a language identification dataset to classify text into
# 20 languages with 99.6% accuracy.
#
# Why use a neural model for language detection?
# - Traditional methods (like langid) use statistical n-gram features
# - Neural models can handle:
#   - Short text better
#   - Mixed-language text
#   - Similar languages (Portuguese vs Spanish)
#   - Code-switching within sentences
#
# =============================================================================


# =============================================================================
# GET ALL LANGUAGE SCORES
# =============================================================================

print("="*60)
print("Full Language Scores")
print("="*60)

# Use top_k=None to get scores for ALL languages
text = "This could be in any language"
results = classifier(text, top_k=None, truncation=True)

print(f"\nText: \"{text}\"")
print(f"\nTop 5 language predictions:")
for i, result in enumerate(results[:5]):
    bar = "█" * int(result['score'] * 30)
    print(f"  {result['label']:4} {result['score']:.2%} {bar}")


# =============================================================================
# PRACTICAL USE CASES
# =============================================================================
#
# Language detection is useful for:
#
# 1. Content routing — Send customer messages to appropriate support teams
# 2. Translation pipelines — Detect source language before translating
# 3. Content moderation — Apply language-specific rules
# 4. Data preprocessing — Filter or categorize multilingual datasets
# 5. User experience — Auto-select UI language
#
# =============================================================================

print("\n" + "="*60)
print("Batch Processing")
print("="*60)

# Process multiple texts at once
texts = [
    "The weather is nice today.",
    "Il fait beau aujourd'hui.",
    "Das Wetter ist heute schön.",
]

results = classifier(texts, truncation=True)

print("\nBatch language detection:\n")
for text, result in zip(texts, results):
    print(f"  \"{text}\"")
    print(f"   → {result['label']} ({result['score']:.1%})\n")