#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check tokenization (OFFLINE) using the Transformers fast tokenizer
loaded from a local Hugging Face model folder.

Supported inputs
----------------
1) Single text:  --text "..."
2) Plain file:   --file /path/to/text.txt
3) JSON / JSONL: --json-file /path/to/data.json[.jsonl/.ndjson]
   Auto-detects one of:
   - NODE MAP: {"node_0": "...", "node_1": "...", ...}
   - EXAMPLES: {"examples":[{"query": <str|[str,...]>, "reference_contexts":[<str>, ...]}, ...]}
   - LEGACY  : array/single objects with "query"/"context" (case-insensitive)
   - JSON-Lines / NDJSON: one object per line; each line may be any of the above.

Choose fields to check (for examples/legacy):
  --json-field {query|reference_contexts|context|both}
  ('context' is an alias of 'reference_contexts')

Scoring
-------
Outputs:
- final_score : combined score (see --score-mode)
- global_score: aggregate risk across the whole string
- max_word_risk: worst local per-word risk (guards against dilution)
- Optional indicators and top-K risky words for explainability

Install
-------
pip install -U transformers
# optional for precise grapheme boundaries (Unicode UAX #29):
pip install regex
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

# Optional precise grapheme support (\X) for Unicode UAX #29
try:
    import regex as regx
except Exception:
    regx = None

# Transformers (required)
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
except Exception:
    AutoTokenizer = None
    PreTrainedTokenizerFast = None


# ---------------- Strict OFFLINE ----------------
def ensure_offline_env(force_offline: bool) -> None:
    if force_offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------- Normalization ----------------
NBSP = "\u00A0"; THINSPACE = "\u2009"; NARROW_NBSP = "\u202F"
EN_SPACE = "\u2002"; EM_SPACE = "\u2003"; FIGURE_SPACE = "\u2007"; HAIR_SPACE = "\u200A"
SOFT_HYPHEN = "\u00AD"; CONTROL_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")

def normalize_text(text: str) -> str:
    # NFC compose; unify CR/CRLF to LF; tame exotic spaces & soft hyphens
    s = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    for ch in [NBSP, THINSPACE, NARROW_NBSP, EN_SPACE, EM_SPACE, FIGURE_SPACE, HAIR_SPACE, "\t"]:
        s = s.replace(ch, " ")
    s = s.replace(SOFT_HYPHEN, "")
    s = CONTROL_RE.sub("", s)
    s = re.sub(r" {2,}", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s


# ---------------- Data structures ----------------
@dataclass
class TokenSpan:
    tid: int
    text: str
    start: int
    end: int  # char indices in the normalized string


# ---------------- HF fast backend ----------------
class HFFastBackend:
    """
    Transformers fast tokenizer (local-only). Returns offset_mapping directly.
    """
    def __init__(self, local_dir: Path, legacy: Optional[bool] = None):
        if AutoTokenizer is None or PreTrainedTokenizerFast is None:
            raise RuntimeError("Transformers not installed. `pip install -U transformers`")
        kwargs = dict(local_files_only=True)
        if legacy is not None:
            kwargs["legacy"] = legacy  # opt-in/out of tokenizer's “legacy behaviour”
        self.tok = AutoTokenizer.from_pretrained(str(local_dir), **kwargs)
        if not isinstance(self.tok, PreTrainedTokenizerFast):
            raise RuntimeError(
                "Loaded tokenizer is not a fast tokenizer. Ensure the local folder includes tokenizer.json."
            )

    def token_spans(self, text: str) -> List[TokenSpan]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans: List[TokenSpan] = []
        for i, (start, end) in enumerate(enc["offset_mapping"]):
            if end >= start:
                spans.append(TokenSpan(i, text[start:end], start, end))
        return spans

    def token_count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


# ---------------- Grapheme boundaries ----------------
def grapheme_boundaries(text: str) -> List[int]:
    """
    Returns positions where a grapheme cluster starts, plus len(text).
    Uses `regex` (\X) if available for precise UAX #29; else a conservative fallback.
    """
    if regx is not None:
        starts = [m.start() for m in regx.finditer(r"\X", text)]
        starts.append(len(text))
        return starts
    # Conservative fallback: start of string and positions after non-combining chars
    starts = [0]
    for i in range(1, len(text)):
        if unicodedata.combining(text[i]) == 0:
            starts.append(i)
    starts.append(len(text))
    return starts


# ---------------- Global indicators & composite score ----------------
def compute_indicators(text: str, spans: List[TokenSpan], token_count: int,
                       frag_word_len_min: int = 8) -> Dict[str, float]:
    Nchars = len(text); Ntok = token_count
    tpc = 100.0 * Ntok / max(1, Nchars)   # tokens per 100 chars
    cpt = Nchars / max(1, Ntok)           # chars per token

    # Token boundaries (internal only)
    tstarts = {s.start for s in spans if s.start not in (0, Nchars)}
    tends   = {s.end   for s in spans if s.end   not in (0, Nchars)}

    # Grapheme split rate
    gstarts = set(grapheme_boundaries(text))
    split_inside = sum(1 for b in tstarts if b not in gstarts)
    gra_split_rate = split_inside / max(1, len(tstarts))

    # Number–unit split rate
    num_unit_re = re.compile(
        r"(\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)"                       # number (+ optional scientific notation)
        r"([A-Za-z%°μΩ/]+(?:[²³\u2070-\u2079\u207A\u207B]|\^[+-]?\d+)?)"  # unit (+ optional superscripts or ^2)
    )

    pairs = list(num_unit_re.finditer(text))
    if pairs:
        junctions = [m.span(1)[1] for m in pairs]
        num_unit_splits = sum(1 for j in junctions if (j in tstarts or j in tends))
        num_unit_split_rate = num_unit_splits / len(pairs)
    else:
        num_unit_split_rate = 0.0

    # Hyphen compound fragmentation
    dash_chars = "-\u2010\u2011\u2012\u2013\u2014"
    hyphen_re = re.compile(rf"[A-Za-z0-9]+(?:[{dash_chars}][A-Za-z0-9]+)+")
    hyphen_spans = list(hyphen_re.finditer(text))
    h_frag_rate = 0.0
    if hyphen_spans:
        over = 0
        for m in hyphen_spans:
            i0 = next((i for i, s in enumerate(spans) if s.end > m.start()), 0)
            i1 = next((i for i, s in enumerate(spans) if s.start >= m.end()), len(spans))
            tok_count_span = max(1, i1 - i0)

            # logical components = hyphens + 1
            hyphens_in_match = sum(1 for c in text[m.start():m.end()] if c in dash_chars)
            comps = hyphens_in_match + 1

            # over-fragmentation beyond minimal (components + possible separator)
            over += max(0, tok_count_span - (comps + 1))

        # Normalize: capped to 1.0; divisor “2.0 * len(hyphen_spans)” mirrors your original scaling
        h_frag_rate = min(1.0, over / (2.0 * len(hyphen_spans)))


    # Prefer Unicode-aware regex if available; else fallback to ASCII
    if regx is not None:
        # Start with an uppercase letter (any script), then one lowercase letter (any script),
        # then allow letters/digits/connector punctuation/apostrophes/hyphens inside.
        proper_re = regx.compile(
            r"\b\p{Lu}\p{Ll}[\p{L}\p{Nd}\p{M}'’_\-]*\b", regx.UNICODE
        )
    else:
        # ASCII fallback 
        proper_re = re.compile(r"\b[A-Z][a-z][A-Za-z0-9'’_-]*\b")

    if regx is not None:
        caps_re = regx.compile(r"\b\p{Lu}{2,}[\p{Lu}\p{Nd}_-]*\b")
    else:
        caps_re = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\b")

    props = list(proper_re.finditer(text)) + list(caps_re.finditer(text))
    ent_frag_rate = 0.0
    if props:
        many = 0
        for m in props:
            i0 = next((i for i, s in enumerate(spans) if s.end > m.start()), 0)
            i1 = next((i for i, s in enumerate(spans) if s.start >= m.end()), len(spans))
            if (i1 - i0) >= 3:
                many += 1
        ent_frag_rate = many / len(props)

    # Morphology/subword fragmentation (fertility of long words)
    # Long-word detection (fertility): Unicode-aware when possible
    if regx is not None:
        word_re = regx.compile(r"\b[\p{L}\p{Nd}][\p{L}\p{Nd}\p{M}'’_\-]*\b", regx.UNICODE)
    else:
        word_re = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’_-]*")
    words = [m for m in word_re.finditer(text) if (m.end() - m.start()) >= frag_word_len_min]
    fert = 0.0
    if words:
        total = 0
        for m in words:
            i0 = next((i for i, s in enumerate(spans) if s.end > m.start()), 0)
            i1 = next((i for i, s in enumerate(spans) if s.start >= m.end()), len(spans))
            total += max(1, i1 - i0)
        fert = total / len(words)

    return {
        "chars": float(Nchars),
        "tokens": float(Ntok),
        "tokens_per_100_chars": float(tpc),
        "chars_per_token": float(cpt),
        "grapheme_split_rate": float(gra_split_rate),
        "number_unit_split_rate": float(num_unit_split_rate),
        "hyphen_compound_frag_rate": float(h_frag_rate),
        "entity_frag_rate": float(ent_frag_rate),
        "long_word_fertility": float(fert),
    }

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def composite_risk_score(ind: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    # Map indicators to 0..1 components with a safe clamp
    r1 = clamp01((ind["tokens_per_100_chars"] - 20.0) / (80.0 - 20.0))    # length inflation
    r2 = clamp01(ind["grapheme_split_rate"])
    r3 = clamp01(ind["number_unit_split_rate"])
    r4 = clamp01(ind["hyphen_compound_frag_rate"])
    r5 = clamp01(ind["entity_frag_rate"])
    fert = ind["long_word_fertility"] if ind["long_word_fertility"] > 0 else 1.0
    r6 = clamp01((fert - 1.0) / (3.0 - 1.0))

    parts = {
        "length_inflation": r1,
        "grapheme_split": r2,
        "number_unit_split": r3,
        "hyphen_frag": r4,
        "entity_frag": r5,
        "morph_fertility": r6,
    }

    wsum = sum(weights.get(k, 0.0) for k in parts)
    if wsum <= 0:
        weights = {
            "length_inflation": 0.25,
            "grapheme_split":  0.20,
            "number_unit_split": 0.15,
            "hyphen_frag":     0.15,
            "entity_frag":     0.10,
            "morph_fertility": 0.15,
        }
        wsum = 1.0

    score = sum(parts[k] * (weights.get(k, 0.0) / wsum) for k in parts)
    return clamp01(score), parts


# ---------------- Local (per-word) risk ----------------
# Local word detection. Unicode-aware when possible.
if regx is not None:
    _WORD_RE = regx.compile(r"\b[\p{L}\p{Nd}][\p{L}\p{Nd}\p{M}'’_\-]*\b", regx.UNICODE)
else:
    _WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’_-]*")

def _word_spans(text: str) -> List[Tuple[int, int, str]]:
    out = []
    for m in _WORD_RE.finditer(text):
        out.append((m.start(), m.end(), text[m.start():m.end()]))
    return out

def _token_span_indices_covering(spans: List[TokenSpan], start: int, end: int) -> Tuple[int, int]:
    i0 = next((i for i, s in enumerate(spans) if s.end > start), 0)
    i1 = next((i for i, s in enumerate(spans) if s.start >= end), len(spans))
    return i0, i1

def _grapheme_split_present(text: str, spans: List[TokenSpan], start: int, end: int) -> bool:
    gstarts = set(grapheme_boundaries(text))
    internal_boundaries = set()
    for s in spans:
        if start < s.start < end:
            internal_boundaries.add(s.start)
        if start < s.end < end:
            internal_boundaries.add(s.end)
    for b in internal_boundaries:
        if b not in gstarts:
            return True
    return False

def _hyphen_over_fragmentation(text_slice: str, tok_count_slice: int) -> float:
    dash_chars = "-\u2010\u2011\u2012\u2013\u2014"
    comps = 1 + sum(1 for c in text_slice if c in dash_chars)
    extra = max(0, tok_count_slice - (comps + 1))
    return min(1.0, extra / 2.0)  # normalize to [0,1]

def compute_word_level_risks(text: str,
                             spans: List[TokenSpan],
                             frag_word_len_min: int = 8) -> List[Dict]:
    words = _word_spans(text)
    results = []
    for (ws, we, wtxt) in words:
        i0, i1 = _token_span_indices_covering(spans, ws, we)
        tok_n = max(1, i1 - i0)
        char_n = max(1, we - ws)

        # 1) local length inflation (tokens/char)
        local_tpc = tok_n / char_n
        r_len = 0.0 if local_tpc <= 0.25 else min(1.0, (local_tpc - 0.25) / (1.0 - 0.25))

        # 2) grapheme split presence
        r_grapheme = 1.0 if _grapheme_split_present(text, spans, ws, we) else 0.0

        # 3) hyphen over-fragmentation
        r_hyphen = _hyphen_over_fragmentation(wtxt, tok_n) if any(
            c in wtxt for c in "-\u2010\u2011\u2012\u2013\u2014") else 0.0

        # 4) entity fragmentation: capitalized word split into >=3 tokens    
        if regx is not None:
            ent_match = bool(regx.match(r"\p{Lu}\p{Ll}.*", wtxt))
        else:
            ent_match = bool(re.match(r"^[A-Z][a-z].*", wtxt))
        r_entity = 1.0 if ent_match and tok_n >= 3 else 0.0

        # 5) fertility for long words
        r_fert = 0.0
        if char_n >= frag_word_len_min:
            r_fert = 0.0 if tok_n <= 1 else min(1.0, (tok_n - 1) / (3 - 1))

        # 6) number–unit junction inside the word
        r_numunit = 0.0
        if re.search(r"\d", wtxt) and re.search(r"[A-Za-z%°μΩohmVA/]", wtxt):
            m = re.search(r"(\d+(?:[.,]\d+)?)([A-Za-z%°μΩohmVA/]+)", wtxt)
            if m:
                junction = ws + m.span(1)[1]
                boundary = any(s.start == junction or s.end == junction for s in spans[i0:i1])
                r_numunit = 1.0 if boundary else 0.0

        wts = {"len": 0.25, "grapheme": 0.25, "hyphen": 0.15, "entity": 0.10, "fert": 0.15, "numunit": 0.10}
        local = (wts["len"] * r_len + wts["grapheme"] * r_grapheme + wts["hyphen"] * r_hyphen +
                 wts["entity"] * r_entity + wts["fert"] * r_fert + wts["numunit"] * r_numunit)

        results.append({
            "word": wtxt, "start": ws, "end": we, "tokens": tok_n,
            "components": {
                "len": round(r_len, 4), "grapheme": r_grapheme, "hyphen": r_hyphen,
                "entity": r_entity, "fert": round(r_fert, 4), "numunit": r_numunit
            },
            "risk": round(min(1.0, max(0.0, local)), 4)
        })
    return results

def combine_scores(global_score: float,
                   max_word_risk: float,
                   mode: str = "both",
                   alpha: float = 0.7,
                   beta: float = 1.0) -> float:
    if mode == "aggregate":
        return global_score
    if mode == "max":
        return max_word_risk
    if mode == "blend" or mode == "both":
        return float(alpha) * global_score + (1.0 - float(alpha)) * max_word_risk
    if mode == "maxof":
        return max(global_score, float(beta) * max_word_risk)
    return max(global_score, max_word_risk)


# ---------------- Audit wrapper ----------------
def audit_sentence(backend: HFFastBackend, text: str, frag_word_len_min: int = 8) -> Dict[str, object]:
    textN = normalize_text(text)
    spans = backend.token_spans(textN)
    token_count = backend.token_count(textN)
    ind = compute_indicators(textN, spans, token_count, frag_word_len_min=frag_word_len_min)
    return {"text": textN, "indicators": ind, "spans": spans}


# ---------------- JSON helpers (NODE MAP + EXAMPLES + LEGACY) ----------------
_NODE_KEY_RE = re.compile(r"^node_(\d+)$")

def _ci_get(d: dict, key: str):
    """Case-insensitive dict get; returns (actual_key, value) or (None, None)."""
    tgt = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == tgt:
            return k, v
    return None, None

def _is_examples_root(data) -> bool:
    if isinstance(data, dict):
        k, v = _ci_get(data, "examples")
        return k is not None and isinstance(v, list)
    return False

def _is_nodes_map(data) -> bool:
    if isinstance(data, dict) and data:
        for k, v in data.items():
            if not (isinstance(k, str) and _NODE_KEY_RE.match(k) and isinstance(v, str)):
                return False
        return True
    return False

def _iter_json_objects_any(path: Path) -> Iterator[dict]:
    """
    Yield normalized objects to process:
      - If NODE MAP: yields {"id": node_key, "node_id": node_key, "query": text, "__nodes_map__": True}
      - If EXAMPLES root: yields each example object as-is
      - If LEGACY array/object: yields dicts as-is
      - If JSON-Lines: same logic per line
    """
    suf = path.suffix.lower()
    if suf in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if _is_nodes_map(obj):
                    # order by numeric suffix for determinism
                    items = sorted(obj.items(), key=lambda kv: int(_NODE_KEY_RE.match(kv[0]).group(1)))
                    for k, v in items:
                        yield {"id": k, "node_id": k, "query": v, "__nodes_map__": True}
                elif _is_examples_root(obj):
                    _, arr = _ci_get(obj, "examples")
                    for ex in arr:
                        if isinstance(ex, dict):
                            yield ex
                elif isinstance(obj, dict):
                    yield obj
                elif isinstance(obj, list):
                    for ex in obj:
                        if isinstance(ex, dict):
                            yield ex
        return

    # Regular .json
    data = json.loads(path.read_text(encoding="utf-8"))
    if _is_nodes_map(data):
        items = sorted(data.items(), key=lambda kv: int(_NODE_KEY_RE.match(kv[0]).group(1)))
        for k, v in items:
            yield {"id": k, "node_id": k, "query": v, "__nodes_map__": True}
        return
    if _is_examples_root(data):
        _, arr = _ci_get(data, "examples")
        for ex in arr:
            if isinstance(ex, dict):
                yield ex
        return
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
        return
    if isinstance(data, dict):
        yield data
        return
    raise ValueError("JSON root must be a node map, an object/array, or an object with 'examples': [ ... ].")

def _extract_strings_from_field(obj: dict, field: str) -> List[Tuple[str, str, int]]:
    """
    Returns [(actual_key, value_str, index_within_field), ...]
    Accepts value as str or list[str]. Ignores non-strings.
    """
    key_used, val = _ci_get(obj, field)
    out: List[Tuple[str, str, int]] = []
    if key_used is None or val is None:
        return out
    if isinstance(val, str):
        if val.strip():
            out.append((key_used, val, 0))
    elif isinstance(val, list):
        j = 0
        for item in val:
            if isinstance(item, str) and item.strip():
                out.append((key_used, item, j)); j += 1
    return out


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Check tokenization (offline) with composite/global and per-word risk."
    )
    ap.add_argument("--local-dir", required=True,
                    help="Filesystem PATH to local HF repo folder (not a Hub ID).")

    # Input modes
    ap.add_argument("--text", help="Sentence/string to audit (if omitted, read stdin unless --json-file).")
    ap.add_argument("--file", help="UTF-8 text file to audit (single combined score).")

    # JSON input
    ap.add_argument("--json-file", help="Path to JSON or JSON-Lines file.")
    ap.add_argument("--json-field",
                    choices=["query", "reference_contexts", "context", "both"],
                    default="query",
                    help="Which field(s) to audit for examples/legacy JSON (default: query). "
                         "'context' aliases 'reference_contexts'. Ignored for node-map input.")
    ap.add_argument("--id-field", default="id",
                    help="Optional field to copy into results if present (default: id).")
    ap.add_argument("--json-max", type=int, default=0,
                    help="If >0, limit the number of objects processed.")
    ap.add_argument("--json-lines", action="store_true",
                    help="Emit JSON-Lines (one JSON result per line) instead of a single JSON object.")

    # Scoring / options
    ap.add_argument("--legacy", choices=["true","false"], default=None,
                    help="If using HF fast tokenizer, set legacy behaviour explicitly.")
    ap.add_argument("--offline", action="store_true", help="Set HF offline env vars.")
    ap.add_argument("--frag-word-len-min", type=int, default=8,
                    help="Min char length for 'long words' in fertility.")
    ap.add_argument("--risk-weights", type=str, default="",
                    help=("JSON dict weights: {length_inflation,grapheme_split,number_unit_split,"
                          "hyphen_frag,entity_frag,morph_fertility}."))
    ap.add_argument("--emit-indicators", action="store_true",
                    help="Include per-indicator values and top-K word risks in output JSON.")

    ap.add_argument("--score-mode",
                    choices=["aggregate", "max", "blend", "maxof", "both"],
                    default="both",
                    help=("How to combine global and local risks. "
                          "'aggregate' = global only; 'max' = worst word only; "
                          "'blend' = alpha*global + (1-alpha)*max_word_risk; "
                          "'maxof' = max(global, beta*max_word_risk); "
                          "'both' = report all and set final to 'blend'."))
    ap.add_argument("--blend-alpha", type=float, default=0.7,
                    help="α in final = α*global + (1-α)*max_word_risk (used in 'blend' and 'both').")
    ap.add_argument("--maxof-beta", type=float, default=1.0,
                    help="β in final = max(global, β*max_word_risk) (used in 'maxof').")
    ap.add_argument("--report-top-k", type=int, default=3,
                    help="Report top-K highest-risk words when --emit-indicators is set.")

    ap.add_argument("--score-only", action="store_true",
                    help="Print only the composite final score(s). "
                         "For node-map JSON and non JSON-Lines, prints a mapping {node_id: score}; "
                         "for JSON-Lines, prints one JSON per line with {'id','final_score'}.")

    # Debugging visibility for input text & tokenization
    ap.add_argument("--emit-text", action="store_true",
        help="Include the normalized input text in outputs.")
    ap.add_argument("--emit-tokens", action="store_true",
        help="Include tokenization with offsets (tokens_debug) in outputs.")

    args = ap.parse_args()

    ensure_offline_env(args.offline)

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.exists():
        print(f"ERROR: --local-dir '{local_dir}' does not exist. Pass a filesystem folder, not a model ID.",
              file=sys.stderr)
        sys.exit(1)

    # Build backend
    legacy = None
    if args.legacy is not None:
        legacy = (args.legacy.lower() == "true")
    backend = HFFastBackend(local_dir, legacy=legacy)

    # Weights
    weights: Dict[str, float] = {}
    if args.risk_weights:
        try:
            weights = json.loads(args.risk_weights)
            assert isinstance(weights, dict)
        except Exception:
            print("WARNING: --risk-weights ignored (must be a JSON dict).", file=sys.stderr)
            weights = {}

    # ------------- Mode 1: JSON input -------------
    if args.json_file:
        jpath = Path(args.json_file).expanduser().resolve()
        if not jpath.exists():
            print(f"ERROR: --json-file '{jpath}' does not exist.", file=sys.stderr)
            sys.exit(1)

        # Normalize field choice for examples/legacy
        field_choice = args.json_field
        if field_choice == "context":
            field_choice = "reference_contexts"
        fields = ["query", "reference_contexts"] if field_choice == "both" else [field_choice]

        max_n = args.json_max if args.json_max and args.json_max > 0 else None

        results: List[dict] = []
        finals_only_list: List[float] = []
        finals_only_map: Dict[str, float] = {}
        obj_index = 0
        nodes_mode = False  # flips to True when we see a node-map normalized object

        for obj in _iter_json_objects_any(jpath):
            if max_n is not None and obj_index >= max_n:
                break

            if obj.get("__nodes_map__"):
                nodes_mode = True

            # Optional ID passthrough
            id_key, id_val = _ci_get(obj, args.id_field)
            # For nodes, we also accept node_id automatically
            if id_key is None:
                id_key, id_val = _ci_get(obj, "node_id")
            id_out = id_val if id_key else None

            # FIELD SELECTION:
            # - In nodes_mode we always audit obj["query"] (already normalized in iterator)
            # - Otherwise we audit selected fields from examples/legacy
            if nodes_mode:
                values = []
                qk, qv = _ci_get(obj, "query")
                if qk and isinstance(qv, str) and qv.strip():
                    values.append((qk, qv, 0))
            else:
                values = []
                for fld in (["query", "reference_contexts"] if field_choice == "both" else [field_choice]):
                    values.extend(_extract_strings_from_field(obj, fld))

            if not values:
                obj_index += 1
                continue

            for (key_used, value, field_item_index) in values:
                # Global audit
                res = audit_sentence(backend, value, frag_word_len_min=args.frag_word_len_min)
                ind = res["indicators"]
                global_score, part_scores = composite_risk_score(ind, weights)

                # Per-word risks
                word_risks = compute_word_level_risks(
                    res["text"], res["spans"], frag_word_len_min=args.frag_word_len_min
                )
                max_word_risk = max((w["risk"] for w in word_risks), default=0.0)

                # Combine
                final = combine_scores(global_score, max_word_risk,
                                       mode=args.score_mode,
                                       alpha=args.blend_alpha,
                                       beta=args.maxof_beta)

                # Score-only streaming
                if args.score_only and args.json_lines:
                    if id_out is not None:
                        print(json.dumps({"id": id_out, "final_score": round(final, 4)}, ensure_ascii=False))
                    else:
                        print(f"{round(final, 4):.4f}")
                elif args.json_lines:
                    out = {
                        "index": obj_index,
                        "field": key_used,
                        "field_item_index": field_item_index,
                        "final_score": round(final, 4),
                        "global_score": round(global_score, 4),
                        "max_word_risk": round(max_word_risk, 4),
                    }
                    if id_out is not None:
                        out["id"] = id_out
                    if args.emit_text:
                        out["text"] = res["text"]
                    if args.emit_tokens:
                        out["tokens_debug"] = [{"id": s.tid, "text": s.text, "start": s.start, "end": s.end}
                                           for s in res["spans"]]
                    if args.emit_indicators:
                        top_k = sorted(word_risks, key=lambda d: d["risk"], reverse=True)[:max(0, args.report_top_k)]
                        out["summary"] = {
                            "chars": int(ind["chars"]),
                            "tokens": int(ind["tokens"]),
                            "tokens_per_100_chars": round(ind["tokens_per_100_chars"], 2),
                            "chars_per_token": round(ind["chars_per_token"], 3),
                        }
                        out["indicators"] = {k: round(float(v), 4) for k, v in ind.items()}
                        out["component_scores"] = {k: round(float(v), 4) for k, v in part_scores.items()}
                        out["top_k_words"] = top_k
                    print(json.dumps(out, ensure_ascii=False))
                else:
                    if args.score_only:
                        if nodes_mode and id_out is not None:
                            finals_only_map[str(id_out)] = round(final, 4)
                        else:
                            finals_only_list.append(round(final, 4))
                    else:
                        out = {
                            "index": obj_index,
                            "field": key_used,
                            "field_item_index": field_item_index,
                            "final_score": round(final, 4),
                            "global_score": round(global_score, 4),
                            "max_word_risk": round(max_word_risk, 4),
                        }
                        if id_out is not None:
                            out["id"] = id_out
                        if args.emit_text:
                            out["text"] = res["text"]
                        if args.emit_tokens:
                            out["tokens_debug"] = [{"id": s.tid, "text": s.text, "start": s.start, "end": s.end}
                                           for s in res["spans"]]
                        if args.emit_indicators:
                            top_k = sorted(word_risks, key=lambda d: d["risk"], reverse=True)[:max(0, args.report_top_k)]
                            out["summary"] = {
                                "chars": int(ind["chars"]),
                                "tokens": int(ind["tokens"]),
                                "tokens_per_100_chars": round(ind["tokens_per_100_chars"], 2),
                                "chars_per_token": round(ind["chars_per_token"], 3),
                            }
                            out["indicators"] = {k: round(float(v), 4) for k, v in ind.items()}
                            out["component_scores"] = {k: round(float(v), 4) for k, v in part_scores.items()}
                            out["top_k_words"] = top_k
                        results.append(out)

            obj_index += 1

        # Batch output (non-JSON-Lines)
        if not args.json_lines:
            if args.score_only:
                if finals_only_map:  # node-map case
                    print(json.dumps(finals_only_map, ensure_ascii=False, indent=2))
                else:
                    print(json.dumps(finals_only_list, ensure_ascii=False, indent=2))
            else:
                print(json.dumps({"results": results, "count": len(results)}, ensure_ascii=False, indent=2))
        return

    # ------------- Mode 2: Plain text / file -------------
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    res = audit_sentence(backend, text, frag_word_len_min=args.frag_word_len_min)
    ind = res["indicators"]
    global_score, part_scores = composite_risk_score(ind, weights)

    # Per-word risks for plain text
    word_risks = compute_word_level_risks(
        res["text"], res["spans"], frag_word_len_min=args.frag_word_len_min
    )
    max_word_risk = max((w["risk"] for w in word_risks), default=0.0)

    final = combine_scores(global_score, max_word_risk,
                           mode=args.score_mode,
                           alpha=args.blend_alpha,
                           beta=args.maxof_beta)

    if args.score_only:
        print(f"{final:.4f}")
        return

    out = {
        "final_score": round(final, 4),
        "global_score": round(global_score, 4),
        "max_word_risk": round(max_word_risk, 4),
        "summary": {
            "chars": int(ind["chars"]),
            "tokens": int(ind["tokens"]),
            "tokens_per_100_chars": round(ind["tokens_per_100_chars"], 2),
            "chars_per_token": round(ind["chars_per_token"], 3),
        },
    }
    if args.emit_text:
        out["text"] = res["text"]
    if args.emit_tokens:
        out["tokens_debug"] = [{"id": s.tid, "text": s.text, "start": s.start, "end": s.end}
                               for s in res["spans"]]
    if args.emit_indicators:
        top_k = sorted(word_risks, key=lambda d: d["risk"], reverse=True)[:max(0, args.report_top_k)]
        out["indicators"] = {k: round(float(v), 4) for k, v in ind.items()}
        out["component_scores"] = {k: round(float(v), 4) for k, v in part_scores.items()}
        out["top_k_words"] = top_k

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
