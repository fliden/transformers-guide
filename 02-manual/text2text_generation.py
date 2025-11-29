#!/usr/bin/env python3
"""
Text-to-Text Generation Without Pipelines
==========================================

This script demonstrates text2text-generation using the manual approach
with T5, an encoder-decoder model. This shows how encoder-decoder
architectures differ from decoder-only models like GPT-2.

Model: T5-small (60M parameters)
Architecture: Encoder-decoder

Key difference from GPT-2:
- GPT-2 (decoder-only): Continues the input sequence
- T5 (encoder-decoder): Transforms input into a different output
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


# =============================================================================
# STEP 1: LOAD MODEL AND TOKENIZER
# =============================================================================
#
# T5ForConditionalGeneration: The T5 model with both encoder and decoder
# - "Conditional" because output is conditioned on the encoded input
# - Different from GPT2LMHeadModel which is decoder-only
#
# T5 comes in several sizes:
# - t5-small: 60M parameters (~240 MB)
# - t5-base: 220M parameters (~890 MB)
# - t5-large: 770M parameters (~3 GB)
# - t5-3b, t5-11b: Even larger

MODEL_NAME = "t5-small"

print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME} ({model.num_parameters():,} parameters)")


# =============================================================================
# STEP 2: UNDERSTAND THE ARCHITECTURE
# =============================================================================
#
# ENCODER-DECODER vs DECODER-ONLY
#
# GPT-2 (Decoder-only):
# ┌─────────────────────────────────┐
# │  "Once upon a time"             │
# │         ↓                       │
# │     [DECODER]                   │
# │         ↓                       │
# │  "Once upon a time, there..."   │  ← Continues the input
# └─────────────────────────────────┘
#
# T5 (Encoder-Decoder):
# ┌─────────────────────────────────┐
# │  "translate English to French:  │
# │   Hello, how are you?"          │
# │         ↓                       │
# │     [ENCODER]                   │  ← Understands the full input
# │         ↓                       │
# │   (hidden representation)       │
# │         ↓                       │
# │     [DECODER]                   │  ← Generates new sequence
# │         ↓                       │
# │  "Bonjour, comment allez-vous?" │  ← Transformed output (not continuation!)
# └─────────────────────────────────┘
#
# The encoder processes the ENTIRE input bidirectionally (can see all tokens).
# The decoder generates output autoregressively, attending to encoder outputs.
#
# =============================================================================


# =============================================================================
# STEP 3: TRANSLATION EXAMPLE
# =============================================================================

print("\n" + "="*60)
print("TRANSLATION")
print("="*60)

# T5 uses task prefixes to know what to do
input_text = "translate English to French: Hello, how are you?"

print(f"\nInput: {input_text}")

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

print(f"\n--- Tokenization ---")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

# Generate output
output_ids = model.generate(
    inputs.input_ids,
    max_length=50,
    num_return_sequences=1,
)

# Decode the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"\n--- Generation ---")
print(f"Output IDs: {output_ids[0]}")
print(f"Output text: {output_text}")

# Notice: Output is ONLY the translation, NOT input + translation
# GPT-2 would give: "translate English to French: Hello... Bonjour..."
# T5 gives just: "Bonjour, comment allez-vous?"


# =============================================================================
# STEP 4: LOOKING AT THE ENCODER-DECODER INTERNALS
# =============================================================================

print("\n" + "="*60)
print("ENCODER-DECODER INTERNALS")
print("="*60)

input_text = "translate English to German: The weather is nice."
inputs = tokenizer(input_text, return_tensors="pt")

# We can examine the encoder and decoder separately
with torch.no_grad():
    # Get encoder outputs (the "understanding" of the input)
    encoder_outputs = model.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )

print(f"Input: {input_text}")
print(f"\nEncoder output shape: {encoder_outputs.last_hidden_state.shape}")
print(f"  - Batch size: {encoder_outputs.last_hidden_state.shape[0]}")
print(f"  - Sequence length: {encoder_outputs.last_hidden_state.shape[1]}")
print(f"  - Hidden dimension: {encoder_outputs.last_hidden_state.shape[2]}")

# The encoder creates a rich representation of the input
# The decoder then uses this to generate the output

# Generate with the full model
output_ids = model.generate(inputs.input_ids, max_length=30)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated output: {output_text}")


# =============================================================================
# STEP 5: MANUAL TOKEN-BY-TOKEN GENERATION
# =============================================================================
#
# Let's see the autoregressive decoding process explicitly.
# This is similar to GPT-2, but the decoder attends to encoder outputs.

print("\n" + "="*60)
print("MANUAL TOKEN-BY-TOKEN GENERATION")
print("="*60)

input_text = "translate English to French: Good morning"
inputs = tokenizer(input_text, return_tensors="pt")

print(f"Input: {input_text}")
print(f"\nGenerating token by token:")

# Start with just the decoder start token
# T5 uses pad_token_id as the decoder start token
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

# Get encoder outputs once (we'll reuse them)
with torch.no_grad():
    encoder_outputs = model.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )

# Generate tokens one at a time
max_tokens = 10
generated_tokens = []

for i in range(max_tokens):
    with torch.no_grad():
        # Decoder forward pass, attending to encoder outputs
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )
    
    # Get logits for the last position
    next_token_logits = outputs.logits[:, -1, :]
    
    # Greedy: pick the highest probability token
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    
    # Check for end of sequence
    if next_token_id.item() == model.config.eos_token_id:
        print(f"  Step {i+1}: <EOS> - stopping")
        break
    
    # Decode and display
    token_text = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token_id.item())
    print(f"  Step {i+1}: '{token_text}' (ID: {next_token_id.item()})")
    
    # Append to decoder input for next iteration
    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(0)], dim=-1)

print(f"\nFinal output: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}")


# =============================================================================
# STEP 6: MULTIPLE TASKS WITH THE SAME MODEL
# =============================================================================

print("\n" + "="*60)
print("ONE MODEL, MANY TASKS")
print("="*60)

tasks = [
    ("Translation", "translate English to French: I love programming."),
    ("Summarization", "summarize: Artificial intelligence is transforming industries worldwide by automating tasks and providing insights from data."),
    ("Q&A", "question: What is the capital of Japan?"),
]

print("\nSame model, different task prefixes:\n")

for task_name, input_text in tasks:
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_length=50,
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"[{task_name}]")
    print(f"  Input:  {input_text[:50]}{'...' if len(input_text) > 50 else ''}")
    print(f"  Output: {output_text}\n")


# =============================================================================
# KEY DIFFERENCES: GPT-2 vs T5
# =============================================================================
#
# | Aspect              | GPT-2 (text-generation)        | T5 (text2text-generation)      |
# |---------------------|--------------------------------|--------------------------------|
# | Architecture        | Decoder-only                   | Encoder-decoder                |
# | Model class         | GPT2LMHeadModel                | T5ForConditionalGeneration     |
# | How it works        | Continues the prompt           | Transforms input to output     |
# | Input processing    | Causal (left-to-right only)    | Bidirectional (encoder)        |
# | Output              | Prompt + continuation          | Just the transformed output    |
# | Best for            | Creative writing, completion   | Translation, summarization     |
# | Prompt style        | Natural continuation           | Task prefix + input            |
#
# When to use which:
#
# Use DECODER-ONLY (GPT-2) for:
#   - Creative writing / story continuation
#   - Code completion
#   - Open-ended chat
#   - When you want the model to "keep going"
#
# Use ENCODER-DECODER (T5) for:
#   - Translation
#   - Summarization
#   - Question answering
#   - Any "transform this into that" task
#
# =============================================================================