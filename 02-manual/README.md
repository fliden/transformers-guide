# Manual Inference with Transformers

This section demonstrates how to use HuggingFace Transformers **without** the high-level pipeline API. Understanding the manual approach helps you grasp what's happening under the hood and gives you more control when you need it.

## Quick Start

```bash
pip install transformers torch
python sentiment_analysis.py
python text_generation.py
```

## What Gets Downloaded?

The first time you load a model, HuggingFace downloads files to a local cache:

| File | Typical Size | Purpose |
|------|--------------|---------|
| `config.json` | ~1 KB | Model architecture settings (layers, hidden size, etc.) |
| `vocab.txt` or `vocab.json` | ~200-500 KB | Vocabulary mapping words/subwords to token IDs |
| `tokenizer_config.json` | ~1 KB | Tokenizer settings (special tokens, etc.) |
| `model.safetensors` | 250MB-3GB+ | The neural network weights |

**Cache location:**

- **macOS/Linux:** `~/.cache/huggingface/hub/`
- **Windows:** `C:\Users\<username>\.cache\huggingface\hub\`

**Managing the cache:**

```bash
# See cache size
du -sh ~/.cache/huggingface/hub/

# Use the HuggingFace CLI
huggingface-cli scan-cache
huggingface-cli delete-cache
```

---

## Core Concepts

### Tokenization

Before a model can process text, it must be converted to numbers:

```python
tokenizer = AutoTokenizer.from_pretrained("model-name")
inputs = tokenizer("Hello world", return_tensors="pt")
```

The tokenizer produces:

- **input_ids**: Integer IDs for each token
- **attention_mask**: Binary mask (1 = real token, 0 = padding)

### What Are Logits?

**Logits** are raw, unnormalized model outputs. They're "confidence scores" before any transformation.

- Can be any real number (positive, negative, zero)
- NOT probabilities (don't sum to 1)
- Higher value = more confident about that option

**For classification** (e.g., sentiment analysis):
```
logits shape: [batch_size, num_classes]
Example: [[-3.5, 4.2]]  →  NEGATIVE score, POSITIVE score
```

**For text generation** (e.g., GPT-2):
```
logits shape: [batch_size, sequence_length, vocab_size]
Example: [1, 4, 50257]  →  For each position, a score for every possible next token
```

### Softmax: Logits → Probabilities

The **softmax** function converts logits to probabilities:

```python
probs = torch.softmax(logits, dim=-1)
```

After softmax:
- All values are between 0 and 1
- All values sum to 1.0
- Relative ordering is preserved

### torch.no_grad()

When making predictions (not training), wrap your code:

```python
with torch.no_grad():
    outputs = model(**inputs)
```

This disables gradient computation, reducing memory usage and speeding up inference.

---

## Text Classification

**Task:** Given text, predict a category (e.g., positive/negative sentiment)

**Model type:** Encoder models with a classification head (e.g., DistilBERT, BERT)

**Process:**
1. Tokenize input text
2. Model produces logits for each class
3. Apply softmax to get probabilities
4. Pick the class with highest probability

```python
# Simplified flow
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probs, dim=-1)
```

See `sentiment_analysis.py` for the full commented example.

---

## Emotion Classification (Multi-Class)

**Task:** Classify text into one of 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)

**Model type:** Same as sentiment - encoder with classification head, but more output classes

**Key insight:** The process is identical to binary classification! The only differences:

| Aspect | Binary (Sentiment) | Multi-Class (Emotion) |
|--------|-------------------|----------------------|
| Logits shape | `[batch, 2]` | `[batch, 7]` |
| Softmax output | 2 probabilities | 7 probabilities |
| Still sums to 1.0 | ✅ | ✅ |

```python
# The code is the same - just more classes
probs = torch.softmax(outputs.logits, dim=-1)  # Now 7 values instead of 2
predicted_class = torch.argmax(probs, dim=-1)   # Still picks the highest
```

**Getting top-k predictions:**

Sometimes you want the top 3 emotions, not just the winner:

```python
top_k = torch.topk(probs[0], k=3)
for score, idx in zip(top_k.values, top_k.indices):
    emotion = model.config.id2label[idx.item()]
    print(f"{emotion}: {score.item():.2%}")
```

See `emotion_classification.py` for the full example.

---

## Language Detection (Multilingual)

**Task:** Identify which language a text is written in (20 languages)

**Model:** XLM-RoBERTa — a multilingual transformer trained on 100 languages

**Key insight:** Same classification process, just with a multilingual model.

### About XLM-RoBERTa

XLM-RoBERTa ("Cross-lingual Language Model") is:

- Trained on 2.5TB of CommonCrawl data in 100 languages
- Learns cross-lingual representations (knowledge transfers between languages)
- A strong foundation for multilingual NLP tasks

The language detection model (papluca/xlm-roberta-base-language-detection) achieves 99.6% accuracy across 20 languages.

### Why Neural Language Detection?

| Aspect | Traditional (langid) | Neural (XLM-RoBERTa) |
|--------|---------------------|----------------------|
| Short text | Struggles | Handles well |
| Similar languages | Often confused | More accurate |
| Model size | ~1 MB | ~1.1 GB |
| Languages | 97 | 20 |
| Accuracy | 98.5% | 99.6% |

### Supported Languages

Arabic (ar), Bulgarian (bg), German (de), Greek (el), English (en), Spanish (es), French (fr), Hindi (hi), Italian (it), Japanese (ja), Dutch (nl), Polish (pl), Portuguese (pt), Russian (ru), Swahili (sw), Thai (th), Turkish (tr), Urdu (ur), Vietnamese (vi), Chinese (zh)

See `language_detection.py` for the full example.

---

## Fill-Mask (Pretraining Objective)

**Task:** Predict what word belongs in the [MASK] position

**Model:** DistilBERT (or BERT) — the BASE model, not fine-tuned

**Key insight:** This is how BERT learns! The pretraining objective.

### How BERT Was Trained

```
Original:  "The cat sat on the mat"
Masked:    "The cat [MASK] on the mat"
Target:    Predict "sat"
```

BERT randomly masks 15% of words and learns to predict them. This teaches the model to "understand" language deeply.

### Bidirectional vs Unidirectional

| Model | Sees Context From | Training Objective |
|-------|-------------------|-------------------|
| BERT | Both left AND right | Predict [MASK] anywhere |
| GPT-2 | Only left | Predict next token |

Why bidirectional matters:

```
"The ___ barked at the mailman."

GPT-2 sees:  "The" → could be anything
BERT sees:   "The" + "barked at the mailman" → probably "dog"
```

### The Pretrain → Fine-tune Pipeline

```
PRETRAINING (fill-mask)          FINE-TUNING (your task)
┌─────────────────────┐          ┌─────────────────────┐
│ Massive text corpus │          │ Add task head       │
│        ↓            │    →     │        ↓            │
│ Predict [MASK]      │          │ Train on labels     │
│        ↓            │          │        ↓            │
│ "Understands" lang  │          │ Sentiment, Q&A, etc │
└─────────────────────┘          └─────────────────────┘
```

This is why DistilBERT can do sentiment, Q&A, NER — they all start from fill-mask!

See `fill_mask.py` for the full example.

---

## Text Generation

**Task:** Given a prompt, continue writing text

**Model type:** Decoder/autoregressive models (e.g., GPT-2, GPT-3, LLaMA)

**Key difference from classification:** Generation is iterative. The model predicts one token at a time, appending each prediction to generate the next.

### The Autoregressive Loop

```
Input: "Once upon a"
         ↓
Model predicts next token → "time"
         ↓
Input: "Once upon a time"
         ↓
Model predicts next token → ","
         ↓
Input: "Once upon a time,"
         ↓
... repeat until max_length or end token
```

### Why Logits Are 3D for Generation

For text generation, logits have shape `[batch, sequence_length, vocab_size]`:

- At position `i`, the model predicts what token comes at position `i+1`
- We typically only need the **last position** (`logits[:, -1, :]`)
- That gives us scores for all 50,257 possible next tokens

### model.generate() Parameters

```python
output_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,  # GPT-2 has no pad token
    max_length=50,                         # Total length (prompt + generated)
    num_return_sequences=1,                # Number of outputs
)
```

### Generation Strategies

How do you pick the next token from 50,257 options?

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Greedy** | Always pick highest probability | Deterministic, can be repetitive |
| **Sampling** | Sample from probability distribution | More creative/varied |
| **Temperature** | Scale logits before sampling (higher = more random) | Control creativity |
| **Top-k** | Sample from top K tokens only | Avoid unlikely tokens |
| **Top-p (nucleus)** | Sample from smallest set with cumulative prob > p | Dynamic candidate pool |
| **Beam search** | Track multiple candidates simultaneously | Tasks with "correct" answers |

```python
# Greedy (default)
model.generate(inputs.input_ids, max_length=50)

# Sampling with temperature
model.generate(inputs.input_ids, max_length=50, do_sample=True, temperature=0.7)

# Top-p sampling
model.generate(inputs.input_ids, max_length=50, do_sample=True, top_p=0.9)
```

See `text_generation.py` for full examples of each strategy.

---

## Text-to-Text Generation (Encoder-Decoder)

**Task:** Transform input text into output text (translation, summarization, Q&A)

**Model type:** Encoder-decoder models (T5, BART, FLAN-T5)

**Key difference from GPT-2:** This is NOT continuation — it's transformation.

### Encoder-Decoder Architecture

```
GPT-2 (Decoder-only):
┌─────────────────────────────────┐
│  "Once upon a time"             │
│         ↓                       │
│     [DECODER]                   │
│         ↓                       │
│  "Once upon a time, there..."   │  ← Continues the input
└─────────────────────────────────┘

T5 (Encoder-Decoder):
┌─────────────────────────────────┐
│  "translate English to French:  │
│   Hello, how are you?"          │
│         ↓                       │
│     [ENCODER]                   │  ← Understands the full input
│         ↓                       │
│   (hidden representation)       │
│         ↓                       │
│     [DECODER]                   │  ← Generates new sequence
│         ↓                       │
│  "Bonjour, comment allez-vous?" │  ← Transformed output (not continuation!)
└─────────────────────────────────┘
```

### Key Differences

| Aspect | text-generation (GPT-2) | text2text-generation (T5) |
|--------|-------------------------|---------------------------|
| Architecture | Decoder-only | Encoder-decoder |
| Model class | `GPT2LMHeadModel` | `T5ForConditionalGeneration` |
| How it works | Continues the prompt | Transforms input to output |
| Input processing | Causal (left-to-right only) | Bidirectional (encoder) |
| Output | Prompt + continuation | Just the transformed output |
| Best for | Creative writing, completion | Translation, summarization |

### The T5 "Text-to-Text" Philosophy

T5 treats ALL NLP tasks as "text in, text out":

```python
# Instead of different model types for each task...
# Just use task prefixes:

"translate English to French: Hello"  → "Bonjour"
"summarize: [long article]"           → "[short summary]"
"question: What is 2+2?"              → "4"
```

This unified approach influenced instruction-following in modern LLMs.

### When to Use Which

**Use decoder-only (GPT-2) for:**
- Creative writing / story continuation
- Code completion
- Open-ended chat
- When you want the model to "keep going"

**Use encoder-decoder (T5) for:**
- Translation
- Summarization
- Question answering
- Any "transform this into that" task

See `text2text_generation.py` for the full example.

---

## Classification vs Generation: Key Differences

| Aspect | Classification | Generation |
|--------|---------------|------------|
| Model type | Encoder (BERT, DistilBERT) | Decoder (GPT-2, LLaMA) |
| Output | Single prediction | Sequence of tokens |
| Logits shape | `[batch, num_classes]` | `[batch, seq_len, vocab_size]` |
| Process | Single forward pass | Iterative (token by token) |
| Post-processing | Softmax + argmax | Decode token IDs to text |

---

## Files in This Section

| File | Model | Task | Key Concepts |
|------|-------|------|--------------|
| `sentiment_analysis.py` | DistilBERT | Sentiment (2 classes) | Logits, softmax, argmax |
| `emotion_classification.py` | DistilRoBERTa | Emotions (7 classes) | Multi-class classification, top-k |
| `language_detection.py` | XLM-RoBERTa | Language (20 classes) | Multilingual models |
| `fill_mask.py` | DistilBERT | Predict [MASK] | Pretraining objective, bidirectional |
| `text_generation.py` | GPT-2 | Text completion | Autoregressive, decoder-only |
| `text2text_generation.py` | T5 | Translation, summarization | Encoder-decoder architecture |

## Next Steps

- Run both scripts and observe the outputs
- Modify the prompts and see how results change
- Try different generation strategies (temperature, top-k, top-p)
- Compare with the pipeline versions in `01-pipelines/`