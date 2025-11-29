# HuggingFace Transformers Guide

A hands-on exploration of the HuggingFace Transformers library, comparing the high-level pipeline API with manual inference approaches.

## Why This Repo?

Most tutorials show you how to use `pipeline()` and call it a day. But understanding what happens under the hood—tokenization, logits, softmax, autoregressive generation—makes you a better ML practitioner. This repo documents my learning journey with working code you can run.

## Repository Structure
```
transformers-guide/
├── 01-pipelines/              # High-level API (quick and easy)
│   ├── sentiment_analysis.py
│   ├── emotion_classification.py
│   ├── language_detection.py
│   ├── fill_mask.py
│   ├── text_generation.py
│   ├── text2text_generation.py
│   └── README.md
│
├── 02-manual/                 # Under the hood (full control)
│   ├── sentiment_analysis.py
│   ├── emotion_classification.py
│   ├── language_detection.py
│   ├── fill_mask.py
│   ├── text_generation.py
│   ├── text2text_generation.py
│   └── README.md              # ← Detailed concept explanations
│
├── MODELS.md                  # Model reference and selection guide
├── README.md
└── pyproject.toml
```

## Quick Start
```bash
# Clone the repo
git clone https://github.com/fliden/transformers-guide
cd transformers-guide

# Install dependencies (uv recommended)
uv sync

# Or using pip
pip install torch transformers sentencepiece

# Run examples
python 02-manual/sentiment_analysis.py
python 02-manual/text_generation.py
```

## What's Covered

| Task | Pipeline | Manual | Architecture | Key Concepts |
|------|----------|--------|--------------|--------------|
| Sentiment Analysis | ✅ | ✅ | Encoder | Binary classification, logits, softmax |
| Emotion Classification | ✅ | ✅ | Encoder | Multi-class, top-k predictions |
| Language Detection | ✅ | ✅ | Encoder (multilingual) | XLM-RoBERTa, 20 languages |
| Fill-Mask | ✅ | ✅ | Encoder | Pretraining objective, bidirectional context |
| Text Generation | ✅ | ✅ | Decoder-only | Autoregressive decoding, sampling strategies |
| Text-to-Text | ✅ | ✅ | Encoder-decoder | Transformation tasks, task prefixes |

## Models

See **[MODELS.md](MODELS.md)** for detailed information about each model, architecture comparisons, hardware requirements, and model selection guidance.

**Quick reference:**
- **Classification**: DistilBERT (sentiment), DistilRoBERTa (emotion), XLM-RoBERTa (language)
- **Generation**: GPT-2
- **Text-to-Text**: T5-small

All models download automatically on first run and are cached locally (~2.7 GB total).

## Key Concepts

| Term | What It Means |
|------|---------------|
| **Tokenizer** | Converts text → token IDs (integers) |
| **Logits** | Raw model output scores (any real number, not probabilities) |
| **Softmax** | Converts logits → probabilities (0-1, sum to 1) |
| **Argmax** | Finds the index of the highest value |
| **Autoregressive** | Generates one token at a time, each depending on previous tokens |
| **Encoder** | Processes input bidirectionally, creates representations (BERT, DistilBERT) |
| **Decoder-only** | Generates text by continuing input (GPT-2, LLaMA) |
| **Encoder-decoder** | Encodes input, then decodes to different output (T5, BART) |
| **Pipeline** | High-level API that wraps all the above |

For detailed explanations, see `02-manual/README.md`.

## Transformer Architectures: A Brief History

Understanding *why* different architectures exist helps you choose the right model:
```
2017: Transformer ("Attention Is All You Need")
      └── Original encoder-decoder for translation
      
2018: GPT (OpenAI)                    2018: BERT (Google)
      └── Decoder-only                      └── Encoder-only
      └── "Predict next token"              └── "Predict masked token"
      └── Good at: generation               └── Good at: understanding
      
2019: GPT-2, DistilBERT, RoBERTa, XLM-RoBERTa...
      └── Bigger models, better training, multilingual
      
2020: T5 (Google) — "Text-to-Text Transfer Transformer"
      └── Back to encoder-decoder
      └── Unified approach: ALL tasks as text-to-text
      └── "translate English to French: Hello" → "Bonjour"
      
2020+: GPT-3, LLaMA, Claude, etc.
      └── Decoder-only dominates for chat/generation
      └── But encoder models still best for classification, embeddings
```

**Which architecture for which task?**

| Task | Best Architecture | Why |
|------|-------------------|-----|
| Classification, NER | Encoder (BERT) | Needs full bidirectional context |
| Text completion, chat | Decoder-only (GPT) | Naturally generates sequences |
| Translation, summarization | Encoder-decoder (T5) | Understands input, generates different output |
| Embeddings, similarity | Encoder (BERT) | Rich bidirectional representations |

## Learning Path

1. **Start with `02-manual/sentiment_analysis.py`** — Binary classification, simplest case
2. **Then `02-manual/emotion_classification.py`** — See that multi-class works the same way
3. **Read `02-manual/README.md`** — Understand logits, softmax, the core concepts
4. **Move to `02-manual/text_generation.py`** — Decoder-only generation (GPT-2 style)
5. **Then `02-manual/text2text_generation.py`** — Encoder-decoder (T5 style), contrast with GPT-2
6. **Compare with `01-pipelines/`** — Appreciate what the pipeline abstracts away
7. **Check `MODELS.md`** — Explore model variants and selection criteria

## Requirements

- Python 3.13+
- PyTorch
- Transformers library
- SentencePiece (for T5 tokenizer)
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install torch transformers sentencepiece
```

## License

MIT