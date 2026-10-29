# Reproduction instructions

All commands are run from the `code/` directory. The supplied data is in `code/data/`
and is not modified.

## 1. Environment

The reference environment used for the reported scores:

| Package | Version |
|---|---|
| Python | 3.11.5 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

Inference (`submission.py`) uses only NumPy and the supplied parser/data. scikit-learn and
scipy are needed only for training.

## 2. Train the frozen checkpoint

```bash
python train_final.py --output runs/final_bundle --C 0.1 --alpha 0.18 --k 80 --beta 0.0
```

This writes:

- `runs/final_bundle/model.npz` — 44,632 bytes
- `runs/final_bundle/config.json` — 577 bytes

Training is deterministic (`random_state=0`, L-BFGS) and takes less than 10 seconds on one
CPU core.

## 3. Evaluate

The reported commands use four threads, matching the resource protocol. On Windows,
set the environment variables in PowerShell before running Python.

### Validation

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python predict.py --method submission.py --checkpoint runs/final_bundle \
  --split validation --output runs/final-validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/final-validation-predictions.json --output runs/final-validation-score.json
```

Expected: `accuracy = 93.23308270676692` (744/798).

### Test

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python predict.py --method submission.py --checkpoint runs/final_bundle \
  --split test --output runs/final-test-predictions.json
python score_predictions.py --split test \
  --predictions runs/final-test-predictions.json --output runs/final-test-score.json
```

Expected: `accuracy = 84.36830835117773` (394/467).

## 4. Frozen artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| `runs/final_bundle/model.npz` | `39D2478B11B51CAEA2EF7539840B41ECFF1A17E7E83831D9464C8884AE4D5B06` |
| `runs/final_bundle/config.json` | `271742AB8A84ABD9F84B8C44D74F9D7B068072C976AE09070FF3A4EF56D44B17` |
| `submission.py` | `B55BD43C33A55350A828B16DFCB7F74ABA8199129D20DEA15CE61890F35249CC` |
| `train_final.py` | `1F793C42D4019CB5770EF5447228737B529B94B25C3B873742B6A2CCC6F2A178` |

The checkpoint hash is the model used for the reported score. Retraining with the command
above should reproduce the same `model.npz`.

## 5. Resource accounting

| Item | Value |
|---|---|
| Checkpoint assets (`model.npz` + `config.json`) | 45,209 bytes |
| Prediction time, full test bank | 1.6–1.8 s (runner); ≈2.3–2.5 s including process start |
| Scoring time | 0.04 s |
| Threads | 4 |
| Test bank | 467 questions |

The 16 MiB asset limit excludes the supplied data. The k-NN verifier reads the supplied
training split (`code/data/candidates/gsm8k_train.npz` and `code/data/gsm8k_train.jsonl`)
at inference; it adds no learned asset beyond the 45 KB checkpoint. If the training split
is unavailable, `submission.py` falls back to the answer-debiased ranker only (82.23% test)
and still accepts a new bank without retraining.

## 6. Ablations (official scorer)

| Checkpoint | Test accuracy |
|---|---:|
| `runs/ablate_A` (`alpha=0`, answer-debiased ranker only) | 82.22698072805139% |
| `runs/ablate_K` (`alpha=1`, k-NN verifier only) | 72.59100642398288% |
| `runs/final_bundle` (`alpha=0.18`) | 84.36830835117773% |

These checkpoints differ only in the blend weight stored in `model.npz`; the underlying
ranker and retrieval data are identical.

## 7. Seeds and data use

- Training seed: `random_state=0` for logistic regression; L-BFGS is deterministic.
- No randomness in inference.
- Training uses only `data/candidates/gsm8k_train.*` and `data/gsm8k_train.jsonl`.
- Validation is used for development and blend-weight selection.
- Test references are used only by the official scorer and for the final reported score;
  they are not read by `train_final.py`, `submission.py`, or the retrieval verifier.
