
# Tokenization Risk Audit (`calculate_tokenization_risk.py`)

A command‑line tool to **audit tokenization offline** using a Hugging Face **fast** tokenizer loaded from a **local** model folder. It computes (1) **global indicators** across the whole string, (2) **per‑word risks**, and (3) an interpretable **final tokenization‑risk score**.

## Highlights
- **Strict offline**: `--offline` sets HF offline env vars and the tokenizer is loaded only from `--local-dir`.
- **Normalization**: NFC composition; unify line breaks; replace exotic spaces; strip soft hyphen and control chars.
- **Grapheme‑aware**: optional `regex` (\X) for Unicode UAX #29; conservative fallback otherwise.
- **Multiple inputs**: single text, file, or JSON/JSONL (node‑map, examples, legacy) with `--json-field` selection.
- **Explainable metrics**: density, grapheme splits, number–unit splits (e.g., `45°C`), hyphen/dash fragmentation, entity fragmentation, long‑word fertility.
- **Per‑word analysis**: six sub‑components and a max‑word guardrail.
- **Final scoring**: `aggregate`, `max`, `blend`, `maxof`, `both` with tunable `--blend-alpha` and `--maxof-beta`.

## Install
```bash
pip install -U transformers
# optional for precise grapheme clusters
pip install regex
```

## Quick Start
```bash
python calculate_tokenization_risk.py   --local-dir /path/to/local/llama3 --offline   --text "Café costs 3.5% at 45°C in state-of-the-art devices"   --emit-indicators
```

## Example
```bash
python calculate_tokenization_risk.py   --local-dir /path/to/local/llama3 --offline   --json-file testset4tokenrisk-calculation.json  --score-only
{
  "node_1": 0.5612,
  "node_2": 0.0944,
  "node_3": 0.5373,
  "node_4": 0.0281,
  "node_5": 0.5119
}
```

## CLI
See `python calculate_tokenization_risk.py -h` for all options. Key flags:
- `--text`, `--file`, `--json-file`, `--json-lines`, `--json-field`, `--json-max`, `--id-field`
- `--frag-word-len-min`, `--risk-weights`, `--emit-indicators`, `--report-top-k`
- `--score-mode {aggregate|max|blend|maxof|both}`, `--blend-alpha`, `--maxof-beta`, `--score-only`

## Output
- **Plain text**: JSON with `final_score`, `global_score`, `max_word_risk`, `summary`; and, with `--emit-indicators`, detailed `indicators`, `component_scores`, `top_k_words`.
- **JSON input**: either JSON-Lines (one per input) or a batch JSON with `{"results": [...], "count": N}`; `--score-only` prints only scores.

## License
This repository is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY‑NC‑SA 4.0)**. See [`LICENSE`](LICENSE).

The example identifiers, product names, model numbers, and technical values in the included test set are synthetic and do not represent real products or confidential data.
