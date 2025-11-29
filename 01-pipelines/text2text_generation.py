#!/usr/bin/env python3
"""
Text-to-Text Generation WITH Pipelines
=======================================

This script demonstrates text2text-generation using encoder-decoder models
like T5. This is fundamentally different from GPT-2 style text generation.

Model: T5-small (60M parameters)
Architecture: Encoder-decoder (not decoder-only like GPT-2)

Key Insight: T5 treats ALL NLP tasks as "text in, text out" transformations.
Translation, summarization, Q&A — they're all just different prompts to the
same model.
"""

from transformers import pipeline

# Create the text-to-text generation pipeline
generator = pipeline(
    "text2text-generation",
    model="t5-small"
)


# =============================================================================
# TEXT-GENERATION VS TEXT2TEXT-GENERATION
# =============================================================================
#
# These are fundamentally different approaches:
#
# | Aspect          | text-generation (GPT-2)      | text2text-generation (T5)      |
# |-----------------|------------------------------|--------------------------------|
# | Architecture    | Decoder-only                 | Encoder-decoder                |
# | How it works    | Continues the prompt         | Transforms input to output     |
# | Use cases       | Creative writing, completion | Translation, summarization, Q&A|
# | Prompt style    | "Once upon a time..."        | "translate English to French:" |
# | Output          | Prompt + generated text      | Just the transformed output    |
#
# GPT-2: "Once upon a time" → "Once upon a time, there was a princess..."
# T5:    "translate English to French: Hello" → "Bonjour"
#
# Notice T5 returns ONLY the answer, not the prompt + answer.
#
# =============================================================================


# =============================================================================
# TRANSLATION
# =============================================================================

print("="*60)
print("TRANSLATION")
print("="*60)

# T5 was trained with specific task prefixes
prompt = "translate English to French: How are you today?"
result = generator(prompt, max_length=50)

print(f"\nInput:  {prompt}")
print(f"Output: {result[0]['generated_text']}")

# More translation examples
translations = [
    "translate English to German: The weather is beautiful.",
    "translate English to French: I love programming.",
    "translate English to Romanian: Where is the train station?",
]

print("\nMore translations:")
for prompt in translations:
    result = generator(prompt, max_length=50)
    print(f"  {prompt}")
    print(f"  → {result[0]['generated_text']}\n")


# =============================================================================
# SUMMARIZATION
# =============================================================================

print("="*60)
print("SUMMARIZATION")
print("="*60)

text = """
The Apollo 11 mission was the first to land humans on the Moon. 
Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed 
the Apollo Lunar Module Eagle on July 20, 1969. Armstrong became 
the first person to step onto the lunar surface six hours later, 
followed by Aldrin. They spent about two and a quarter hours 
together outside the spacecraft.
"""

prompt = f"summarize: {text}"
result = generator(prompt, max_length=50)

print(f"\nOriginal text: {text.strip()[:100]}...")
print(f"\nSummary: {result[0]['generated_text']}")


# =============================================================================
# QUESTION ANSWERING (Closed-book)
# =============================================================================
#
# T5 can answer questions from its training knowledge (closed-book QA)
# or from provided context (reading comprehension).

print("\n" + "="*60)
print("QUESTION ANSWERING")
print("="*60)

questions = [
    "question: What is the capital of France?",
    "question: Who wrote Romeo and Juliet?",
    "question: What is the largest planet in our solar system?",
]

print("\nClosed-book Q&A (from model's knowledge):")
for prompt in questions:
    result = generator(prompt, max_length=20)
    print(f"  Q: {prompt.replace('question: ', '')}")
    print(f"  A: {result[0]['generated_text']}\n")


# =============================================================================
# GRAMMAR CORRECTION
# =============================================================================

print("="*60)
print("GRAMMAR CORRECTION")
print("="*60)

# T5 can also do grammar correction with the right prefix
# (though specialized models exist for this)
sentences = [
    "grammar: He go to school yesterday.",
    "grammar: She don't like apples.",
]

# Note: t5-small may not be great at this - larger models work better
print("\nGrammar correction attempts:")
for prompt in sentences:
    result = generator(prompt, max_length=30)
    print(f"  Input:  {prompt.replace('grammar: ', '')}")
    print(f"  Output: {result[0]['generated_text']}\n")


# =============================================================================
# WHY ENCODER-DECODER?
# =============================================================================
#
# Encoder-decoder models have two parts:
#
# 1. ENCODER: Reads and "understands" the entire input
#    - Processes all input tokens at once
#    - Creates a rich representation of the input
#    - Has full bidirectional attention (can see all tokens)
#
# 2. DECODER: Generates the output based on the encoder's representation
#    - Generates one token at a time (autoregressive)
#    - Can attend to all encoder outputs
#    - Has causal attention (only sees previous output tokens)
#
# This is powerful for TRANSFORMATION tasks:
#    - Translation: Understand the whole sentence, then translate
#    - Summarization: Understand the whole document, then summarize
#    - Q&A: Understand the question, then answer
#
# Decoder-only models (GPT-2) are better for CONTINUATION tasks:
#    - Creative writing: Keep going from where you left off
#    - Code completion: Continue the code
#    - Chat: Continue the conversation
#
# =============================================================================


# =============================================================================
# THE T5 "TEXT-TO-TEXT" PHILOSOPHY
# =============================================================================
#
# T5 (Text-to-Text Transfer Transformer) was a landmark paper that showed:
#
# "Every NLP task can be framed as text-to-text"
#
# Instead of:
#   - Classification model with softmax over classes
#   - Translation model with separate encoder/decoder
#   - Q&A model with span extraction
#
# Just use ONE model that takes text and outputs text:
#   - Classification: "classify: I love this movie" → "positive"
#   - Translation: "translate English to French: Hello" → "Bonjour"
#   - Q&A: "question: What is 2+2?" → "4"
#
# This unified approach influenced later models like FLAN-T5, mT5, and
# contributed to the instruction-following capabilities in modern LLMs.
#
# =============================================================================

print("="*60)
print("ONE MODEL, MANY TASKS")
print("="*60)

tasks = [
    ("Translation", "translate English to French: Good morning"),
    ("Summarization", "summarize: Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
    ("Q&A", "question: What color is the sky?"),
]

print("\nSame model (t5-small), different tasks:\n")
for task_name, prompt in tasks:
    result = generator(prompt, max_length=30)
    print(f"  [{task_name}]")
    print(f"  Input:  {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"  Output: {result[0]['generated_text']}\n")