# Calibration Datasets

Curated subsets of open datasets used for threshold calibration and benchmarking.

## Files

| File | Source | License | Records | Purpose |
|------|--------|---------|---------|---------|
| `fever_subset.jsonl` | [FEVER](https://fever.ai/) | CC-BY-SA 3.0 | 200 claims | Contradiction detection calibration |
| `scitail_subset.jsonl` | [SciTail](https://allenai.org/data/scitail) (AllenAI) | Apache 2.0 | 200 pairs | Dependency classification calibration |
| `truthfulqa_subset.jsonl` | [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Apache 2.0 | 100 questions | Adversarial testing |
| `arc_sentences.jsonl` | [ARC Corpus](https://allenai.org/data/arc) (AllenAI) | CC-BY-SA 4.0 | 200 sentences | Seeding + extraction accuracy |

## Format

All files use JSONL format (one JSON object per line).

### fever_subset.jsonl
```json
{"claim": "The boiling point of water is 100 degrees Celsius.", "label": "SUPPORTS", "id": 12345}
```
Labels: `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`
Distribution: 100 SUPPORTS, 50 REFUTES, 50 NOT ENOUGH INFO

### scitail_subset.jsonl
```json
{"premise": "Water freezes at 0 degrees Celsius.", "hypothesis": "Water becomes ice at 0 degrees.", "label": "entails"}
```
Labels: `entails`, `neutral`
Distribution: 100 entails, 100 neutral

### truthfulqa_subset.jsonl
```json
{"question": "What happens if you eat watermelon seeds?", "best_answer": "The seeds pass through your digestive system", "correct_answers": ["Nothing happens", "..."], "incorrect_answers": ["You grow watermelons in your stomach", "..."], "category": "Misconceptions"}
```

### arc_sentences.jsonl
```json
{"text": "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.", "domain": "science"}
```

## How These Were Created

Each file is a stratified random sample from the full dataset, created with `random.seed(42)`.
The one-time subsetting scripts are not checked in — only the resulting JSONL files are committed.

## Benchmark Results

Benchmark output JSON files are written to `results/` (gitignored).
