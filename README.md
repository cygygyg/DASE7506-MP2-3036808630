# DASE7506 MP2 — Selecting Reliable LLM Reasoning

**Student ID:** `[FILL IN]`

**Final official test score:** **84.3683%** (394/467)
**Official validation score:** **93.2331%** (744/798)

**Method:** within-question blend of

1. a linear ranker on per-question-centred candidate embeddings, after removing
   directions explained by the candidate's numeric answer (`answer-debiased ranker`), and
2. a k-NN verifier over the supplied training split that compares a candidate's
   reasoning residual with correct/incorrect prototypes of similar training questions.

The blend is `0.82 * z(A) + 0.18 * z(K)`, with scores standardised within each question.
Candidates with unparseable or non-integer final answers are excluded (all reference
answers are integers). No test references are used for training, retrieval, tuning or
model selection.

## Layout

| Path | Purpose |
|---|---|
| `report.md` | Project report (method, experiments, ablations, analysis, reproduction). |
| `REPRODUCE.md` | Exact install / train / evaluate commands, environment and hashes. |
| `code/submission.py` | Frozen inference (`predict(bank, arrays, checkpoint)`), pure NumPy. |
| `code/train_final.py` | Trains the answer-debiased ranker and writes the checkpoint. |
| `code/runs/final_bundle/model.npz` | Frozen checkpoint (44,632 bytes). |
| `code/runs/final-test-score.json` | Official scorer output for the submitted test predictions. |
| `code/runs/final-validation-score.json` | Official scorer output for validation. |
| `code/exp_*.py`, `code/diag_*.py` | Exploratory scripts used for the report; not needed for reproduction. |

## Quick reproduction

```bash
cd code
python -m pip install -r requirements.txt
python check_data.py
python train_final.py --output runs/final_bundle
python predict.py --method submission.py --checkpoint runs/final_bundle \
  --split test --output runs/final-test-predictions.json
python score_predictions.py --split test \
  --predictions runs/final-test-predictions.json --output runs/final-test-score.json
```

Expected test score: **84.3683%**. Validation: **93.2331%**. See `REPRODUCE.md` for the
full commands, resource measurements and checkpoint hash.

## AI and reuse disclosure

AI coding assistance (an AI coding agent) was used to implement and run experiments,
refactor the final code, and help draft the report. The author reviewed the protocol,
verified the official scores and is responsible for the final method and claims. No
external datasets, pretrained weights or code beyond the supplied package were used.
