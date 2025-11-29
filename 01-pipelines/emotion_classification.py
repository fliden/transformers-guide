#!/usr/bin/env python3
"""
Emotion Classification WITH Pipelines
======================================

This script shows multi-class emotion classification using the pipeline API.
Unlike binary sentiment (positive/negative), this model classifies text into
7 distinct emotions.

Model: j-hartmann/emotion-english-distilroberta-base
Classes: anger, disgust, fear, joy, neutral, sadness, surprise
"""

from transformers import pipeline

# Create the emotion classification pipeline
# Note: We use "text-classification" (not "sentiment-analysis") because
# this is a general classification task, not just positive/negative sentiment
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # Return scores for ALL classes, not just the top one
)

# Single prediction
text = "I can't believe they surprised me with a birthday party!"

print(f"Text: {text}")
print(f"\nEmotion scores:")

results = classifier(text)
# top_k=None returns a list of lists, so we access [0]
for emotion in results[0]:
    bar = "█" * int(emotion['score'] * 20)  # Simple bar visualization
    print(f"  {emotion['label']:8} {emotion['score']:.2%} {bar}")


# =============================================================================
# COMPARING DIFFERENT EMOTIONS
# =============================================================================

print("\n" + "="*60)
print("Emotion Examples")
print("="*60)

examples = [
    "I'm so happy I could dance!",
    "This is absolutely disgusting behavior.",
    "I'm terrified of what might happen next.",
    "I can't believe they would do this to me!",
    "The meeting is scheduled for 3pm tomorrow.",
    "I miss my grandmother so much.",
    "Wait, you're getting married?!",
]

for text in examples:
    results = classifier(text)
    # Get the top emotion (highest score)
    top_emotion = max(results[0], key=lambda x: x['score'])
    print(f"\n\"{text}\"")
    print(f"  → {top_emotion['label']} ({top_emotion['score']:.1%})")


# =============================================================================
# SENTIMENT VS EMOTION
# =============================================================================
#
# Sentiment Analysis (binary):
#   - POSITIVE or NEGATIVE
#   - Good for: product reviews, feedback polarity
#
# Emotion Classification (multi-class):
#   - anger, disgust, fear, joy, neutral, sadness, surprise
#   - Good for: understanding nuanced reactions, customer service, UX research
#
# The same text can have clear emotion but ambiguous sentiment:
#   "I can't believe this happened!" → surprise (emotion), but positive or negative?
#
# =============================================================================