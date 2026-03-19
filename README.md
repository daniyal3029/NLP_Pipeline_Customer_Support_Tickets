# Hybrid Semantic Retrieval & Intelligence System (HSRIS)

> Assignment 3 — NLP Pipeline for Customer Support Ticket Retrieval  
> Built entirely from scratch using **PyTorch** and **NumPy** — no sklearn, no HuggingFace.

---

## Overview

HSRIS is a multi-stage NLP pipeline that processes ~8,470 customer support tickets and retrieves the most relevant ones for any given query. It combines classical statistical methods (TF-IDF) with neural semantic embeddings (GloVe) into a weighted hybrid search engine, deployed as an interactive Gradio app.

---

## Pipeline Architecture

```
Raw Ticket Text
      │
      ▼
┌─────────────────────────────┐
│   Preprocessing             │
│   • Placeholder replacement │
│   • Log/noise removal       │
│   • Regex tokenization      │
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌─────────────────┐
│  TF-IDF  │  │  GloVe 200-d    │
│  Sparse  │  │  Sentence Vec   │
│  Tensor  │  │  (TF-IDF wtd.)  │
└────┬─────┘  └──────┬──────────┘
     │               │
     └──────┬────────┘
            ▼
   FinalScore = 0.4 × TF-IDF + 0.6 × GloVe
            │
            ▼
       Top-5 Results
```

---

## Features

### Part 1 — Categorical Encoders
| Encoder | Field | Detail |
|---|---|---|
| Label Encoding | Ticket Priority | Low=0, Medium=1, High=2, Critical=3. Unseen → -1 |
| One-Hot Encoding | Ticket Channel | Binary vector per channel. Unseen category → zero vector |

### Part 2 — Sparse Keyword Retrieval
- **Custom Tokenizer** — regex-based, lowercasing, removes noise (logs, timestamps, IPs, prices, placeholders)
- **Bag of Words** — top 5,000 token vocabulary, count vectors
- **N-Gram Generator** — sliding window bigrams and trigrams (e.g. `"not_working"`, `"payment_not_working"`)
- **TF-IDF** — smooth IDF (`log((N+1)/(df+1)) + 1`) computed from scratch, stored as `torch.sparse` tensor to save Kaggle RAM

### Part 3 — Dense Semantic Layer
- **GloVe 200-d** — loaded into `torch.nn.Embedding` with frozen weights
- **OOV Strategy** — out-of-vocabulary tokens mapped to zero vector
- **TF-IDF Weighted Averaging** — rare technical keywords (e.g. `billing`, `kernel`) weighted higher than common words, preventing semantic dilution

### Hybrid Search
```
FinalScore = α · TF-IDF_Score + (1 − α) · GloVe_Score     [α = 0.4]
```

### Performance Optimization
- Dual T4 GPU support via `torch.nn.DataParallel`
- Batch similarity computed with `torch.mm(F.normalize(Q), F.normalize(DB).T)`
- Timed over batch sizes: 10, 25, 50, 75, 100 queries
- Execution time vs batch size plotted and saved

### Evaluation
- **Precision@5** reported for TF-IDF, GloVe, and Hybrid across 12 query/type pairs
- **5 qualitative examples** demonstrating GloVe outperforming TF-IDF on semantic paraphrases

### Gradio App
- Free-text ticket description input
- **α slider** (0.0 → 1.0) to shift between keyword and semantic matching live
- Displays predicted **Ticket Type** (top-5 majority vote)
- Shows **top-3 similar past tickets** with subject, description, and resolution

---

## Extra Work (Beyond Assignment Requirements)

| Extra | Description |
|---|---|
| `build_display_text()` | Full noise-removal pipeline — replaces `{product_purchased}` with real product name, strips log lines, timestamps, IPs, prices, camelCase/snake_case tokens, and filters generic filler sentences |
| Smooth IDF | Uses sklearn-style `log((N+1)/(df+1)) + 1` instead of plain `log(N/df)` for better robustness |
| `display_results()` | Formatted ranked output showing Score, Type, Subject, Description for every result |
| Search wrappers | Clean `tfidf_search_str`, `glove_search_str`, `hybrid_search_str` API accepting raw query strings |
| `predict_ticket_type()` | Top-5 majority vote classifier for ticket type prediction |
| Gradio Examples panel | 5 pre-filled example queries for instant demo |
| Summary table | Markdown documentation of every pipeline component |

---

## Dataset

- **Source:** [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset)
- **Scale:** ~8,470 records
- **Fields used:** Ticket Description, Ticket Subject, Ticket Priority, Ticket Type, Ticket Channel, Product Purchased, Resolution

---

## Setup

### Platform
Run on [Kaggle](https://www.kaggle.com/) with **GPU T4 x2** accelerator.

### Datasets required (add to Kaggle notebook)
1. `waseemalastal/customer-support-ticket-dataset`
2. `rtatman/glove-global-vectors-for-word-representation` (200-d)

### Dependencies
All pre-installed on Kaggle. Gradio installed at runtime:
```python
pip install gradio
```

---

## Project Structure

```
HSRIS/
│
├── notebook.ipynb          # Main Kaggle notebook (all code)
├── README.md               # This file
├── batch_timing.png        # Execution time vs batch size plot (generated)
│
└── Links
    ├── GitHub              # https://github.com/your-username/hsris
    ├── Medium Article      # https://medium.com/@your-username/hsris
    └── LinkedIn Post       # https://linkedin.com/in/your-username
```

---

## Results

| Metric | TF-IDF | GloVe | Hybrid (α=0.4) |
|---|---|---|---|
| Mean Precision@5 | — | — | — |

*Fill in after running on Kaggle.*

---

## Assignment Info

- **Course:** Data Science for Software Engineering
- **Assignment:** 3 — Hybrid Semantic Retrieval & Intelligence System
- **Team Size:** Max 2 members
- **Platform:** Kaggle (Dual T4 x2 GPU)
- **Constraint:** No sklearn, no HuggingFace — PyTorch + NumPy only
