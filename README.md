# InstrumentalEval × OLMo-3

Evaluates **instrumental convergence behaviors** across OLMo-3's post-training checkpoints (SFT, DPO, RLVR + step-wise RLVR snapshots) on the InstrumentalEval benchmark (He et al., 2025 — arXiv:2502.12206).

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | All experiment knobs + secret loading (env vars / `.env`) |
| `functions.py` | All function definitions: model loading, inference, judges, metrics, INLP |
| `evaluations.py` | End-to-end driver: discovers steps → runs eval → analyzes results |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for the secrets file (copy to `.env`) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # then fill in your keys
```

Required secrets:
- `OPENAI_API_KEY` — GPT-4o judge
- `GEMINI_API_KEY` — only if running the Gemini second judge (`--gemini`)
- `HF_TOKEN` — optional; speeds up HuggingFace downloads

## Run

```bash
# Full pipeline
python evaluations.py

# Quick sanity check on one task before committing to the full eval
python evaluations.py --sanity-check

# Re-analyze previously-saved results without re-running inference
python evaluations.py --skip-eval

# Add Gemini second judge + IAR analysis
python evaluations.py --gemini
```

Edit `config.py` to change model size, checkpoints, RLVR step subsampling, or
output paths. Anything that was set in the original notebook's "Section 0"
lives there.

## Results

All artifacts (response caches, judge verdicts, summary CSVs) land in
`OUTPUT_DIR` (defaults to `./results`).
