# Hybrid Semantic Retrieval & Intelligence System (HSRIS)

> Assignment 3 — NLP Pipeline for Customer Support Ticket Retrieval  
> Built entirely from scratch using **PyTorch** and **NumPy** — no sklearn, no HuggingFace.

## Overview

HSRIS is a multi-stage NLP pipeline that processes ~8,470 customer support tickets and retrieves the most relevant ones for any given query. It combines classical statistical methods (TF-IDF) with neural semantic embeddings (GloVe) into a weighted hybrid search engine, deployed as an interactive Gradio app.

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
## Features

### Part 1 — Categorical Encoders
| Encoder | Field | Detail |
|---|---|---|
| Label Encoding | Ticket Priority | Low=0, Medium=1, High=2, Critical=3. Unseen → -1 |
| One-Hot Encoding | Ticket Channel | Binary vector per channel. Unseen category → zero vector |

### Part 2 — Sparse Keyword Retrieval
- **Custom Tokenizer** — regex-based, lowercasing, removes noise (logs, timestamps, IPs, prices, placeholders)
- **Bag of Words** — top 5,000 token vocabulary, count vectors
- **N-Gram Generator** — sliding window bigrams and trigrams (e.g. `not_working`, `payment_not_working`)
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
- Shows results with explanation

## Getting Started
To get started with the project, follow these steps:
1. Clone the repository: `git clone https://github.com/daniyal3029/NLP_Pipeline_Customer_Support_Tickets.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the Gradio app: `python app.py`

## Contributing
Contributions are welcome! To contribute to the project, please follow these steps:
1. Fork the repository: `git fork https://github.com/daniyal3029/NLP_Pipeline_Customer_Support_Tickets.git`
2. Create a new branch: `git branch my-branch`
3. Make your changes and commit them: `git commit -m "my changes"`
4. Push your changes to the remote repository: `git push origin my-branch`
5. Create a pull request: `git pull-request`
