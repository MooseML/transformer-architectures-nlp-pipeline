# Transformers Project: Architecture, Preprocessing, NER, and QA

This project brings together four transformer-focused assignments into a unified, modular NLP pipeline. It demonstrates both theoretical understanding and practical application of Transformer-based models.

---

## Project Structure

```
notebooks/             <- Jupyter notebooks for each project phase
src/                   <- Modular Python code for models, data, and training
data/                  <- Static data (NER + QA datasets, GloVe)
models/                <- Pretrained/fine-tuned model checkpoints
tokenizers/            <- Task-specific tokenizer files
results/               <- TensorBoard logs and output directories
tests/                 <- Unit tests for CI/CD
assets/                <- Visuals (diagrams, architecture drawings)
```

---

## Included Tasks

### 0. Transformer Architecture (Custom Implementation)
- Built from scratch with positional encoding, self-attention, multi-head attention.

### 1. Preprocessing & Embedding
- Visualizing and applying GloVe + sinusoidal positional encodings.

### 2. Named Entity Recognition
- Fine-tunes a pretrained transformer on NER data using Hugging Face.

### 3. Question Answering
- Implements both PyTorch and TensorFlow pipelines for extractive QA.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/transformers_project.git
cd transformers_project

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

---

## CI/CD

Unit tests are included and designed to run automatically on GitHub Actions. Additional test cases can be added to `tests/`.

---

## Contributing

Pull requests welcome. For major changes, open an issue first.

---

## Credits

This code was built and modified as part of an Deep Learning specialization course combining four projects:
- Transformer Architecture
- Preprocessing & Positional Encoding
- Named Entity Recognition
- Question Answering

```

---
