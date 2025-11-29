# Pipelines: The Easy Way

The `pipeline()` API is HuggingFace's highest-level abstraction. It handles tokenization, inference, and post-processing in a single function call.

## Quick Start

```bash
pip install transformers torch sentencepiece  # sentencepiece needed for T5
python sentiment_analysis.py
python emotion_classification.py
python language_detection.py
python fill_mask.py
python text_generation.py
python text2text_generation.py
```

## Sentiment Analysis (Binary Classification)

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
result = classifier("I love this movie!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Emotion Classification (Multi-Class)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # Return all class scores
)
result = classifier("I'm so happy!")
# [{'label': 'joy', 'score': 0.98}, {'label': 'surprise', 'score': 0.01}, ...]
```

## Language Detection (Multilingual)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)
result = classifier("Bonjour, comment ça va?")
# [{'label': 'fr', 'score': 0.9998}]
```

This model detects 20 languages with 99.6% accuracy using XLM-RoBERTa, a multilingual transformer trained on 100 languages.

## Fill-Mask (Pretraining Objective)

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")
result = fill_mask("The capital of France is [MASK].")
# [{'token_str': 'paris', 'score': 0.89}, ...]
```

This is how BERT-style models are trained! They learn language by predicting masked words, then get fine-tuned for downstream tasks like classification.

## Text Generation (Decoder-Only)

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
# [{'generated_text': 'Once upon a time, there was a...'}]
```

## Text-to-Text Generation (Encoder-Decoder)

```python
from transformers import pipeline

generator = pipeline("text2text-generation", model="t5-small")
result = generator("translate English to French: Hello, how are you?")
# [{'generated_text': 'Bonjour, comment allez-vous?'}]
```

## text-generation vs text2text-generation

These use fundamentally different architectures:

| Aspect | text-generation (GPT-2) | text2text-generation (T5) |
|--------|-------------------------|---------------------------|
| Architecture | Decoder-only | Encoder-decoder |
| How it works | Continues the prompt | Transforms input to output |
| Use cases | Creative writing, completion | Translation, summarization, Q&A |
| Prompt style | "Once upon a time..." | "translate English to French: Hello" |
| Output | Prompt + generated text | Just the transformed output |

**Use decoder-only (GPT-2) for:** Creative writing, code completion, open-ended chat

**Use encoder-decoder (T5) for:** Translation, summarization, question answering, any "transform this into that" task

## sentiment-analysis vs text-classification

These are actually the same pipeline! `sentiment-analysis` is just an alias for `text-classification`.

Use `sentiment-analysis` when your task is clearly about sentiment (positive/negative).
Use `text-classification` for other classification tasks (emotions, topics, spam, etc.).

## Available Pipelines

| Task | Pipeline Name | Example Use |
|------|---------------|-------------|
| Sentiment Analysis | `sentiment-analysis` | Positive/negative classification |
| Text Classification | `text-classification` | Any classification (emotions, topics, etc.) |
| Fill Mask | `fill-mask` | Predict [MASK] tokens (BERT pretraining) |
| Text Generation | `text-generation` | Continue a prompt (GPT-2 style) |
| Text-to-Text | `text2text-generation` | Transform input to output (T5 style) |
| Translation | `translation_en_to_fr` | Language translation |
| Summarization | `summarization` | Condense long text |
| Question Answering | `question-answering` | Extract answers from context |
| Named Entity Recognition | `ner` | Find people, places, organizations |

## When to Use Pipelines

**Good for:**
- Quick prototyping
- Standard tasks with sensible defaults
- When you just need the final result

**Consider the manual approach when:**
- You need access to raw logits
- Custom generation strategies
- Batch processing with specific requirements
- Learning how transformers work

## Files

| File | Task | Model | Architecture |
|------|------|-------|--------------|
| `sentiment_analysis.py` | Sentiment (2-class) | DistilBERT | Encoder |
| `emotion_classification.py` | Emotion (7-class) | DistilRoBERTa | Encoder |
| `language_detection.py` | Language (20-class) | XLM-RoBERTa | Encoder (multilingual) |
| `fill_mask.py` | Predict masked word | DistilBERT | Encoder (pretraining task) |
| `text_generation.py` | Text completion | GPT-2 | Decoder-only |
| `text2text_generation.py` | Translation, summarization | T5 | Encoder-decoder |

## See Also

`02-manual/` — The same tasks without pipeline abstractions, showing what happens under the hood.

---

## Pipeline Function Reference

```python
from transformers import pipeline

pipe = pipeline(
    task,                    # Required: "text-classification", "fill-mask", etc.
    model=None,              # Model name or path (uses default if None)
    tokenizer=None,          # Usually auto-detected from model
    device=None,             # -1 for CPU, 0+ for GPU, "mps" for Apple Silicon
    **kwargs                 # Task-specific options
)
```

**Common parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `task` | The NLP task to perform | `"text-classification"` |
| `model` | Model identifier from HuggingFace Hub | `"distilbert-base-uncased"` |
| `device` | Where to run inference | `0` (GPU), `-1` (CPU), `"mps"` |
| `top_k` | Number of results to return | `5` or `None` (all) |
| `truncation` | Truncate long inputs | `True` |

**Task-specific parameters (passed to the pipeline call):**

| Task | Parameter | Description |
|------|-----------|-------------|
| Text generation | `max_length` | Maximum output length |
| Text generation | `do_sample` | Enable sampling vs greedy |
| Text generation | `temperature` | Randomness (higher = more random) |
| Text generation | `num_return_sequences` | Number of outputs |
| Classification | `top_k` | Return top K predictions |

**Supported tasks:**

| Task String | Description |
|-------------|-------------|
| `text-classification` | Classify text into categories |
| `sentiment-analysis` | Alias for text-classification |
| `fill-mask` | Predict [MASK] tokens |
| `text-generation` | Continue a prompt (GPT-2 style) |
| `text2text-generation` | Transform input to output (T5 style) |
| `question-answering` | Extract answers from context |
| `summarization` | Condense long text |
| `translation_xx_to_yy` | Translate between languages |
| `ner` / `token-classification` | Named entity recognition |
| `zero-shot-classification` | Classify without training data |
| `feature-extraction` | Get model embeddings |