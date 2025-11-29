#!/usr/bin/env python3
"""
Text Generation Without Pipelines
==================================

This script demonstrates how to generate text using HuggingFace Transformers
WITHOUT the high-level pipeline API. This gives you full control over the
generation process and helps you understand what's happening under the hood.

Model: GPT-2 (124M parameters, smallest version)
Task: Causal language modeling (predict the next token)

GPT-2 is an autoregressive model - it generates text one token at a time,
where each new token depends on all previous tokens.

Prerequisites:
    pip install transformers torch
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# =============================================================================
# STEP 1: LOAD THE MODEL AND TOKENIZER
# =============================================================================
#
# GPT2Tokenizer: Converts text to token IDs using Byte Pair Encoding (BPE)
#   - BPE breaks words into subword units (e.g., "unhappiness" → "un", "happiness")
#   - Vocabulary size: 50,257 tokens
#   - Can handle any text (no "unknown" tokens needed)
#
# GPT2LMHeadModel: The GPT-2 model with a "language modeling head"
#   - The "head" is a linear layer that predicts the next token
#   - "LM" = Language Model
#   - Without the head (GPT2Model), you'd get embeddings but no predictions
#
# Note: "gpt2" is an alias for "openai-community/gpt2" on the Hub

MODEL_NAME = "gpt2"

print("Loading tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME} ({model.num_parameters():,} parameters)")


# =============================================================================
# STEP 2: PREPARE THE PROMPT
# =============================================================================
#
# For text generation, we provide a "prompt" - the starting text that the
# model will continue. The tokenizer converts this to input_ids.
#
# Unlike classification, we typically DON'T need attention_mask for simple
# generation with a single prompt (no padding needed).

prompt = "Once upon a time"

print(f"\nPrompt: {prompt}")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\n--- Tokenization Results ---")
print(f"Input IDs: {inputs['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")


# =============================================================================
# STEP 3: GENERATE TEXT (Using model.generate())
# =============================================================================
#
# The generate() method handles the autoregressive loop for you:
#   1. Run the model on current tokens → get logits for next token
#   2. Select the next token (various strategies available)
#   3. Append the new token to the sequence
#   4. Repeat until max_length or end-of-sequence token
#
# Key parameters:
#
# input_ids: Starting token sequence
#
# attention_mask: Which tokens to attend to (1=real, 0=padding)
#   - Important when batching multiple prompts of different lengths
#   - For single prompts without padding, can be omitted
#
# pad_token_id: Token ID used for padding
#   - GPT-2 doesn't have a native pad token, so we use eos_token_id
#   - This prevents warnings about padding
#
# max_length: Maximum total length (prompt + generated tokens)
#   - Set this to control output length
#   - GPT-2's maximum context is 1024 tokens
#
# num_return_sequences: How many different outputs to generate
#   - Each will be different if using sampling (do_sample=True)
#   - All will be identical with greedy decoding (do_sample=False, default)

print("\n--- Generating Text ---")

output_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,  # Avoid padding warnings
    max_length=50,                         # Total length including prompt
    num_return_sequences=1,                # Number of outputs
)

print(f"Output shape: {output_ids.shape}")  # [num_sequences, sequence_length]
print(f"Output IDs: {output_ids[0][:10]}... (first 10 tokens)")


# =============================================================================
# STEP 4: DECODE THE OUTPUT
# =============================================================================
#
# Convert token IDs back to human-readable text.
#
# skip_special_tokens=True removes tokens like <|endoftext|> from output

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"\n--- Generated Text ---")
print(generated_text)


# =============================================================================
# ALTERNATIVE: MANUAL TOKEN-BY-TOKEN GENERATION
# =============================================================================
#
# The generate() method is convenient, but let's see what happens under the hood.
# This manual approach shows the autoregressive process explicitly.
#
# At each step:
#   1. Model outputs logits for ALL positions
#   2. We only care about the LAST position (next token prediction)
#   3. We pick a token (greedy = highest logit, or sample from distribution)
#   4. Append it and repeat

print("\n" + "="*60)
print("Manual Token-by-Token Generation (Educational)")
print("="*60)

# Start with the prompt
current_ids = inputs.input_ids.clone()
max_new_tokens = 20

print(f"\nGenerating {max_new_tokens} tokens manually...")
print(f"Starting tokens: {tokenizer.decode(current_ids[0])}")

for i in range(max_new_tokens):
    # Forward pass - get logits for all positions
    with torch.no_grad():
        outputs = model(current_ids)
    
    # outputs.logits shape: [batch_size, sequence_length, vocab_size]
    # We only need the logits for the LAST token position
    next_token_logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
    
    # Greedy selection: pick the token with highest logit
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    # Append the new token to our sequence
    current_ids = torch.cat([current_ids, next_token_id], dim=-1)
    
    # Show progress (optional)
    new_token = tokenizer.decode(next_token_id[0])
    if i < 5:  # Show first 5 tokens
        print(f"  Token {i+1}: '{new_token}' (ID: {next_token_id.item()})")

print(f"\nManually generated text:")
print(tokenizer.decode(current_ids[0], skip_special_tokens=True))


# =============================================================================
# UNDERSTANDING LOGITS IN TEXT GENERATION
# =============================================================================
#
# For text generation, logits have shape [batch, sequence_length, vocab_size]
#
# At each position i, the model predicts what token should come at position i+1.
# The logits at position i are a score for each of the 50,257 possible tokens.
#
# Example: If prompt is "Once upon a" (3 tokens):
#   - logits[:, 0, :] → predicts token after "Once"
#   - logits[:, 1, :] → predicts token after "Once upon"
#   - logits[:, 2, :] → predicts token after "Once upon a" ← This is what we want!
#
# This is why we use outputs.logits[:, -1, :] - we want the prediction
# for what comes AFTER the last token in our sequence.

print("\n" + "="*60)
print("Logits Shape Explained")
print("="*60)

with torch.no_grad():
    outputs = model(inputs.input_ids)

print(f"Input: '{prompt}' ({inputs.input_ids.shape[1]} tokens)")
print(f"Logits shape: {outputs.logits.shape}")
print(f"  - Batch size: {outputs.logits.shape[0]}")
print(f"  - Sequence length: {outputs.logits.shape[1]}")
print(f"  - Vocabulary size: {outputs.logits.shape[2]}")

# What does the model predict after "Once upon a time"?
last_position_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
top_5_tokens = torch.topk(last_position_logits, 5)

print(f"\nTop 5 predicted next tokens after '{prompt}':")
for score, token_id in zip(top_5_tokens.values, top_5_tokens.indices):
    token = tokenizer.decode(token_id)
    prob = torch.softmax(last_position_logits, dim=0)[token_id].item()
    print(f"  '{token}' (ID: {token_id.item()}, logit: {score:.2f}, prob: {prob:.2%})")


# =============================================================================
# GENERATION STRATEGIES
# =============================================================================
#
# Different ways to select the next token:
#
# 1. GREEDY (default): Always pick highest probability token
#    - Deterministic (same output every time)
#    - Can be repetitive
#
# 2. SAMPLING (do_sample=True): Sample from probability distribution
#    - More varied/creative outputs
#    - temperature controls randomness (higher = more random)
#
# 3. TOP-K SAMPLING: Sample from top K most likely tokens only
#    - Avoids very unlikely tokens
#
# 4. TOP-P (NUCLEUS) SAMPLING: Sample from smallest set of tokens
#    whose cumulative probability exceeds P
#    - Dynamically adjusts candidate pool
#
# 5. BEAM SEARCH (num_beams > 1): Keep track of multiple candidates
#    - Better for tasks with "correct" answers
#    - Slower but can find better sequences

print("\n" + "="*60)
print("Generation Strategies Comparison")
print("="*60)

# Greedy (deterministic)
greedy_output = model.generate(
    inputs.input_ids,
    max_length=30,
    pad_token_id=tokenizer.eos_token_id,
)
print(f"\nGreedy decoding:")
print(f"  {tokenizer.decode(greedy_output[0], skip_special_tokens=True)}")

# Sampling with temperature
sampling_output = model.generate(
    inputs.input_ids,
    max_length=30,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)
print(f"\nSampling (temperature=0.7):")
print(f"  {tokenizer.decode(sampling_output[0], skip_special_tokens=True)}")

# Top-k sampling
topk_output = model.generate(
    inputs.input_ids,
    max_length=30,
    do_sample=True,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)
print(f"\nTop-k sampling (k=50):")
print(f"  {tokenizer.decode(topk_output[0], skip_special_tokens=True)}")

# Top-p (nucleus) sampling
topp_output = model.generate(
    inputs.input_ids,
    max_length=30,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
print(f"\nTop-p sampling (p=0.9):")
print(f"  {tokenizer.decode(topp_output[0], skip_special_tokens=True)}")