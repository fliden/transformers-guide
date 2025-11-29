#!/usr/bin/env python3
"""
Text Generation WITH Pipelines
==============================

This script shows text generation using the high-level pipeline API.
Compare this with 02-manual/text_generation.py to see what the pipeline
abstracts away.

The pipeline handles:
- Loading the tokenizer and model
- Tokenizing your prompt
- Running the generation loop
- Decoding the output tokens

All in ~5 lines of code.
"""

from transformers import pipeline, set_seed

# Create the text generation pipeline
# Explicitly specify the model to avoid warnings
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Set seed for reproducibility (generation involves randomness by default)
set_seed(42)

# Generate text from a prompt
prompt = "Once upon a time"
results = generator(
    prompt,
    max_length=50,           # Total length including prompt
    num_return_sequences=1,  # How many outputs to generate
)

print(f"Prompt: {prompt}")
print(f"\nGenerated text:")
print(results[0]['generated_text'])


# =============================================================================
# MULTIPLE OUTPUTS
# =============================================================================

print("\n" + "="*60)
print("Multiple Sequences")
print("="*60)

set_seed(42)
results = generator(
    "The meaning of life is",
    max_length=40,
    num_return_sequences=3,
)

for i, result in enumerate(results, 1):
    print(f"\nOutput {i}:")
    print(f"  {result['generated_text']}")


# =============================================================================
# CONTROLLING GENERATION
# =============================================================================
#
# You can pass generation parameters directly to the pipeline:
#
# - do_sample: Enable sampling (vs greedy decoding)
# - temperature: Higher = more random (default 1.0)
# - top_k: Only sample from top K tokens
# - top_p: Only sample from tokens with cumulative prob < p

print("\n" + "="*60)
print("Different Generation Settings")
print("="*60)

# More creative (higher temperature)
set_seed(42)
creative = generator(
    "Once upon a time",
    max_length=40,
    do_sample=True,
    temperature=1.2,
)
print(f"\nCreative (temperature=1.2):")
print(f"  {creative[0]['generated_text']}")

# More focused (lower temperature)
set_seed(42)
focused = generator(
    "Once upon a time",
    max_length=40,
    do_sample=True,
    temperature=0.5,
)
print(f"\nFocused (temperature=0.5):")
print(f"  {focused[0]['generated_text']}")


# =============================================================================
# That's it!
#
# But now you know what's happening under the hood:
#   1. Prompt is tokenized into input_ids
#   2. Model predicts logits for the next token
#   3. A token is selected (greedy, sampling, etc.)
#   4. New token is appended and process repeats
#   5. Output tokens are decoded back to text
#
# See 02-manual/text_generation.py for the full breakdown.
# =============================================================================