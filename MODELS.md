# Models Reference

Detailed information about the pre-trained models used in this repository. Each model is explored in both `01-pipelines/` (high-level API) and `02-manual/` (implementation details).

## Text Classification Models

### DistilBERT (SST-2 Fine-tuned)
- **Model ID**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Task**: Binary sentiment analysis (positive/negative)
- **Architecture**: Encoder-only (distilled BERT)
- **Parameters**: 66M
- **Download Size**: ~268 MB
- **Training Data**: Stanford Sentiment Treebank v2 (SST-2)
- **Output**: 2 classes (NEGATIVE, POSITIVE)
- **Key Feature**: Fast inference, good accuracy for sentiment tasks
- **Examples**: 
  - `01-pipelines/sentiment_analysis.py`
  - `02-manual/sentiment_analysis.py`
- **Notes**: Distilled from BERT-base, 40% smaller, 60% faster, retains 97% of performance

---

### DistilRoBERTa (Emotion Classification)
- **Model ID**: `j-hartmann/emotion-english-distilroberta-base`
- **Task**: Multi-class emotion classification
- **Architecture**: Encoder-only (distilled RoBERTa)
- **Parameters**: 82M
- **Download Size**: ~330 MB
- **Output**: 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Key Feature**: Fine-grained emotion detection beyond simple sentiment
- **Examples**:
  - `01-pipelines/emotion_classification.py`
  - `02-manual/emotion_classification.py`
- **Notes**: Based on RoBERTa (optimized BERT), shows multi-class classification patterns

---

### XLM-RoBERTa (Language Detection)
- **Model ID**: `papluca/xlm-roberta-base-language-detection`
- **Task**: Language identification
- **Architecture**: Encoder-only (multilingual)
- **Parameters**: 278M
- **Download Size**: ~1.1 GB
- **Output**: 20 languages (ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh)
- **Key Feature**: Cross-lingual understanding, trained on 100 languages
- **Examples**:
  - `01-pipelines/language_detection.py`
  - `02-manual/language_detection.py`
- **Notes**: Demonstrates multilingual encoder models, larger than English-only variants

---

### DistilBERT (Base Uncased)
- **Model ID**: `distilbert-base-uncased`
- **Task**: Masked language modeling (fill-mask)
- **Architecture**: Encoder-only
- **Parameters**: 66M
- **Download Size**: ~256 MB
- **Training Objective**: Predict masked tokens in text
- **Key Feature**: Shows the original BERT pretraining task
- **Examples**:
  - `01-pipelines/fill_mask.py`
  - `02-manual/fill_mask.py`
- **Notes**: Not fine-tuned for downstream tasks, uses bidirectional context

## Text Generation Models

### GPT-2
- **Model ID**: `gpt2` (base model)
- **Task**: Autoregressive text generation
- **Architecture**: Decoder-only (causal language model)
- **Parameters**: 124M (base), variants up to 1.5B (XL)
- **Download Size**: ~548 MB
- **Training**: Next-token prediction on web text
- **Key Feature**: Demonstrates autoregressive generation, sampling strategies
- **Examples**:
  - `01-pipelines/text_generation.py`
  - `02-manual/text_generation.py`
- **Generation Parameters**:
  - `max_new_tokens`: How many tokens to generate
  - `temperature`: Randomness (lower = more deterministic)
  - `top_k`: Sample from top-k tokens
  - `top_p`: Nucleus sampling threshold
  - `do_sample`: Enable/disable sampling
- **Notes**: Classic decoder-only architecture, foundation for understanding modern LLMs

## Text-to-Text Models

### T5-Small
- **Model ID**: `t5-small`
- **Task**: Text-to-text transformation (translation, summarization, Q&A)
- **Architecture**: Encoder-decoder
- **Parameters**: 60M
- **Download Size**: ~240 MB
- **Training**: Multi-task text-to-text framework
- **Key Feature**: Unified approach - all tasks framed as text-to-text
- **Task Prefixes**:
  - Translation: `"translate English to French: "`
  - Summarization: `"summarize: "`
  - Q&A: `"question: ... context: ..."`
- **Examples**:
  - `01-pipelines/text2text_generation.py`
  - `02-manual/text2text_generation.py`
- **Tokenizer**: SentencePiece (subword tokenization)
- **Notes**: Encoder processes input, decoder generates output - best for transformation tasks

---

## Architecture Comparison

| Architecture | Models in This Repo | Best For | How It Works |
|--------------|---------------------|----------|--------------|
| **Encoder-only** | DistilBERT, DistilRoBERTa, XLM-RoBERTa | Classification, NER, embeddings | Bidirectional attention, processes full input |
| **Decoder-only** | GPT-2 | Text generation, completion, chat | Causal attention, generates left-to-right |
| **Encoder-decoder** | T5 | Translation, summarization, transformation | Encoder understands input, decoder generates output |

## Model Selection Guide

**Choose encoder models (BERT-family) when:**
- You need to understand/classify existing text
- Full context matters (past and future words)
- Tasks: sentiment analysis, NER, question answering, similarity

**Choose decoder models (GPT-family) when:**
- You need to generate new text
- Continuation/completion is the goal
- Tasks: writing assistance, code generation, chatbots

**Choose encoder-decoder models (T5, BART) when:**
- Input and output are different
- You're transforming text (not just classifying or continuing)
- Tasks: translation, summarization, paraphrasing

## Download and Caching

All models are automatically downloaded on first use and cached locally:
- **macOS/Linux**: `~/.cache/huggingface/hub/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\`

Models persist across runs - you only download once.

## Hardware Requirements

All models in this repo run on:
- **CPU**: All models work, slower inference
- **Apple Silicon (M1/M2/M3)**: MPS acceleration available for PyTorch
- **NVIDIA GPU**: CUDA acceleration (not required for these small models)

For M2 MacBook specifically:
```python
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
```

## Model Variants

Many models come in different sizes. This repo uses smaller variants for learning:

| Model Family | Variants | This Repo Uses |
|--------------|----------|----------------|
| GPT-2 | base (124M), medium (355M), large (774M), xl (1.5B) | base (124M) |
| T5 | small (60M), base (220M), large (770M), 3B, 11B | small (60M) |
| BERT | base (110M), large (340M) | DistilBERT (66M) |

Larger variants generally perform better but require more memory and compute.

## Further Reading

- **HuggingFace Model Hub**: https://huggingface.co/models
- **Model Cards**: Each model on HF has documentation explaining training, use cases, limitations
- **Transformers Docs**: https://huggingface.co/docs/transformers