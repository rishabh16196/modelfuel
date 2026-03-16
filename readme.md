# DistillStack

Cognitive ETL Pipeline for synthesizing grounded instruction-tuning datasets from complex PDFs. Designed to feed Small Language Models (Phi-4, Gemma-2, etc.) with high-fidelity training data.

## Architecture

```
PDF ──► DocumentProcessor (Docling) ──► InternalDocument
              │                              │
              ▼ (low confidence / scanned)   ▼
         VLM Fallback               SynthesisAgent (LiteLLM)
         (DeepSeek-OCR 2)                    │
                                             ▼
                                      QualityScorer (Critic)
                                             │
                                             ▼
                                    Scored JSONL output
```

**Pipeline stages:**

1. **DocumentProcessor** — Extracts structural data from PDFs via IBM Docling. Scanned or low-confidence pages are routed to a VLM fallback.
2. **SynthesisAgent** — Chunks extracted Markdown by logical headers, then uses a teacher LLM to generate grounded Instruction → Thought → Response triples.
3. **QualityScorer** — A critic LLM audits each record for faithfulness (is it grounded in the source?) and complexity (is it useful for SLM training?). Only passing records are emitted.

## Setup

Requires Python 3.12+.

```bash
# Clone and enter the repo
git clone <repo-url> && cd modelfuel

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### CLI

```bash
# Extract structured data from a PDF
distillstack extract path/to/document.pdf

# Specify a custom output directory
distillstack extract path/to/document.pdf --output-dir ./results
```

This produces two files in the output directory:
- `<name>.md` -- extracted Markdown
- `<name>.json` -- full structured `InternalDocument` (pages, content blocks, metadata)

### API Server

```bash
uvicorn distillstack.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path                | Description                                      |
|--------|---------------------|--------------------------------------------------|
| GET    | `/health`           | Liveness check                                   |
| POST   | `/extract`          | Upload a PDF, receive structured JSON             |
| POST   | `/extract/markdown` | Upload a PDF, receive extracted Markdown           |

```bash
# Get structured JSON
curl -X POST http://localhost:8000/extract \
  -F "file=@document.pdf" -o result.json

# Get Markdown only
curl -X POST http://localhost:8000/extract/markdown \
  -F "file=@document.pdf" -o result.md
```

## Configuration

All settings are managed via environment variables (prefixed with `DISTILL_`) or a `.env` file. See `.env.example` for the full list.

| Variable                          | Default              | Description                        |
|-----------------------------------|----------------------|------------------------------------|
| `DISTILL_LITELLM_API_KEY`        | —                    | API key for LiteLLM calls          |
| `DISTILL_TEACHER_MODEL`          | `gpt-4o-mini`        | Model used for synthesis           |
| `DISTILL_CRITIC_MODEL`           | `gpt-4o-mini`        | Model used for quality scoring     |
| `DISTILL_VLM_MODEL`             | `deepseek/deepseek-ocr-2` | VLM for scanned page fallback |
| `DISTILL_VLM_CONFIDENCE_THRESHOLD` | `0.85`            | Pages below this route to VLM      |
| `DISTILL_MAX_CHUNK_TOKENS`       | `1024`               | Max tokens per synthesis chunk     |
| `DISTILL_OUTPUT_DIR`             | `output`             | Default output directory           |
| `DISTILL_LOG_LEVEL`             | `INFO`               | Logging verbosity                  |

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Lint and format
ruff check src/
ruff format src/

# Run tests
pytest
```

## Project Structure

```
src/distillstack/
├── __init__.py
├── config.py              # Pydantic-settings configuration
├── cli.py                 # CLI entry point
├── models/
│   ├── __init__.py
│   └── document.py        # Domain models (InternalDocument, SynthesisRecord, etc.)
├── pipeline/
│   ├── __init__.py
│   ├── processor.py       # DocumentProcessor (Docling + VLM router)
│   ├── synthesis.py       # SynthesisAgent (grounded pair generation)
│   └── quality.py         # QualityScorer (critic audit)
└── api/
    ├── __init__.py
    └── app.py             # FastAPI application
```
