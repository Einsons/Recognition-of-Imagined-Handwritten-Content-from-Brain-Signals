# Extended Experimental Report

## Recognition of Imagined Handwritten Content from EEG

This document records the complete reproduction and model-development study performed on this repository. It complements the original [`README.md`](README.md), which describes the initial architectures and originally reported results.

The study had four goals:

1. reproduce the repository in an isolated environment;
2. determine whether the reported performance was reproducible;
3. evaluate substantially different EEG architectures rather than only tuning the original CNNs;
4. construct a stronger ensemble without directly optimizing weights on test labels.

The best checkpoint-based result obtained in this study is:

| Metric | Accuracy |
|---|---:|
| Validation accuracy | **23.33%** |
| Held-out test accuracy | **24.36%** |

The final prediction is produced by a five-model logit ensemble containing three EEGNet variants, a time-window DeepConvNet, and a graph-based EEG model.

---

## 1. Dataset

The dataset contains single-trial EEG recorded from one participant while imagining handwriting one of the 26 English letters.

| Property | Value |
|---|---:|
| Trials | 7,800 |
| Classes | 26 |
| Trials per class | 300 |
| EEG channels | 24 |
| Samples per trial | 801 |
| Time range | -200 to 3,000 ms |
| Sampling frequency | 250 Hz |
| Chance accuracy | 3.85% |

The MATLAB tensor has shape `(24, 801, 7800)`. The extraction script transposes it to `(7800, 24, 801)` and converts labels from `1..26` to `0..25`.

The supplied data has already undergone artifact removal, baseline correction, and 0.1--45 Hz band-pass filtering. Inspection nevertheless showed that trials were not numerically standardized to identical variance. Trial standard deviations varied approximately from 6.2 to 15.0 across the 1st--99th percentiles. This motivated explicit normalization experiments, although they did not improve the strongest models.

Channel order:

```text
Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8
T7  T8  P7 P8 Fz Cz Pz M1 M2 AFz CPz POz
```

---

## 2. Reproduction Environment

| Component | Version / device |
|---|---|
| Architecture | ARM64 (`aarch64`) |
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GB10 |
| NumPy | 2.4.0 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |

Activate the repository-local environment:

```bash
source .venv/bin/activate
```

---

## 3. Data Splitting and Evaluation Policy

### Primary chronological split

| Split | Trials per class | Total trials |
|---|---:|---:|
| Training | 240 | 6,240 |
| Validation | 30 | 780 |
| Test | 30 | 780 |

For every letter, the first 80% of trials are used for training, the next 10% for validation, and the final 10% for testing. This is stricter than random splitting because it exposes temporal distribution shift caused by fatigue, impedance changes, adaptation, and acquisition drift.

Models and ensemble weights were selected using validation predictions. Failed models were not promoted because of a favorable test result. Repeated experiments nevertheless create indirect familiarity with the fixed test set, so differences of one or two trials must be interpreted cautiously. A publication-quality follow-up should reserve a new untouched test set or use additional participants.

---

## 4. Reproduced Baseline

| Model | Test accuracy |
|---|---:|
| Logistic regression | 13.72% |
| DeepConvNet | 18.46% |
| EEGNet k=15 with SWA | 21.54% |
| DeepConvNet + EEGNet | 22.56% |
| DeepConvNet + EEGNet with nominal BN adaptation | 22.56% |
| EEGNet k=25 | 18.72% |
| Original three-model ensemble | **23.21%** |

The logistic-regression accuracy exactly matched the original report, supporting the correctness of the downloaded data and split. Deep-model performance was lower than the highest value in the original README, which is plausible because EEG optimization is sensitive to CUDA kernels, framework versions, initialization order, and random-number consumption.

### Batch-normalization adaptation

The original `adapt_bn_stats()` updated BatchNorm statistics and then restored the old values, so it behaved like no adaptation. True adaptation reduced validation accuracy from 20.90% to 19.87%. It was therefore excluded. With only 780 target trials, global target statistics can erase useful training-domain information without improving class-conditional alignment.

---

## 5. New Model Experiments

### 5.1 Residual Temporal Convolutional Network

File: [`models/eeg_res_tcn.py`](models/eeg_res_tcn.py)

The model uses four temporal kernels `(7, 15, 31, 63)`, residual depthwise temporal blocks, GroupNorm, and dilations `(1, 2, 4)`.

| Variant | Validation accuracy |
|---|---:|
| Global mean/max pooling | 3.97% |
| Sixteen retained temporal bins | 6.54% |

Global pooling assumes translation invariance. That is inappropriate for stimulus-locked ERP decoding: a response at 200 ms and the same response at 1,500 ms do not have the same physiological meaning. Retaining temporal bins helped but remained far below specialized EEG CNNs.

### 5.2 ShallowConvNet / FBCSP

File: [`models/shallow_conv_net.py`](models/shallow_conv_net.py)

This model uses temporal filtering, full spatial convolution, squaring, average pooling, and logarithmic compression.

| Result | Accuracy |
|---|---:|
| Best observed validation accuracy | approximately 9.10% |

The square--pool--log operations emphasize band power. Imagined handwriting in this dataset appears to depend more on precisely time-locked ERP structure than stationary oscillatory power.

### 5.3 Fixed Filter-Bank network

File: [`models/filter_bank_net.py`](models/filter_bank_net.py)

The signal was decomposed into 1--4, 4--8, 8--13, 13--30, and 30--45 Hz FIR bands, then fused with learned temporal blocks.

| Result | Accuracy |
|---|---:|
| Best validation accuracy | 6.54% |

The result agrees with ShallowConvNet: explicit stationary band power is not the dominant discriminative signal for these 26 classes.

### 5.4 EEG Conformer

Files: [`models/eeg_conformer.py`](models/eeg_conformer.py) and [`src/experiment_conformer.py`](src/experiment_conformer.py)

The model combines a convolutional EEG tokenizer and a three-layer Transformer encoder. CLS-token aggregation and full-token flattening were both tested.

| Variant | Validation accuracy |
|---|---:|
| CLS-token aggregation | approximately 3.85% |
| Full-token flattening | approximately 3.85% |

A diagnostic test reached 100% training accuracy on 104 balanced examples after 50 steps. The implementation was functional, but the approximately one-million-parameter attention model memorized rather than generalized. Strong local ERP priors were more useful than generic capacity.

### 5.5 GraphEEGNet

Files: [`models/graph_eeg_net.py`](models/graph_eeg_net.py) and [`src/experiment_graph_eeg.py`](src/experiment_graph_eeg.py)

GraphEEGNet represents the 24 electrodes as a graph built from approximate 10--20 scalp coordinates. Its normalized graph is:

```text
A_norm = D^(-1/2) (A + I) D^(-1/2)
```

A graph-temporal block performs gated neighbor propagation:

```text
H' = H + TemporalConv(H + sigmoid(g) * A_norm H)
```

| Property | Value |
|---|---:|
| Parameters | 192,570 |
| Input window | 0--2,000 ms |
| Best-checkpoint validation accuracy | 16.54% |
| Highest single-epoch validation accuracy | 18.08% |

Nearby electrodes observe correlated cortical fields, so graph propagation is physiologically motivated. GraphEEGNet was not the strongest individual model, but its errors were sufficiently different to improve the final ensemble.

---

## 6. Temporal-Window Experiments

| Input window | Parameters | Best-checkpoint validation accuracy |
|---|---:|---:|
| Full -200--3,000 ms | 278,246 | 14.87--15.64% depending on seed |
| 0--1,500 ms | 168,006 | 17.56% |
| 0--2,000 ms | 199,206 | **18.08%** |

Removing the prestimulus period and final second increased signal-to-noise ratio and reduced model size. Early activity contains visual encoding, orthographic processing, motor planning, and imagined movement. The 1,500--2,000 ms segment remained useful, while 2,000--3,000 ms appeared predominantly noisy.

---

## 7. Additional Training and Inference Experiments

### Multi-seed ensemble

Independent seed-42 and seed-123 models produced partially decorrelated errors. The first expanded ensemble reached 22.44% validation and 24.10% test accuracy.

### Time-shift test-time augmentation

Symmetric shifts of `±1`, `±2`, `±4`, and `±8` samples all reduced validation accuracy. The learned representations depend on stimulus-locked latency, so inference-only temporal shifts misalign ERP components.

### Logistic stacking

Five models produced 130 concatenated logits. A regularized multinomial meta-classifier selected by internal five-fold validation achieved 18.08% internal CV and 23.08% test accuracy. With only 780 meta-training samples and 26 classes, learned class-specific interactions overfit more than simple fixed weights.

### Temperature calibration

| Fusion method | Best validation accuracy |
|---|---:|
| Raw weighted logits | **23.33%** |
| Temperature-calibrated logits | 23.08% |
| Temperature-calibrated probabilities | 22.95% |

Calibration changed confidence but did not improve top-1 accuracy.

### EEGNet k=7

The 28 ms EEGNet achieved 16.28% best-checkpoint validation accuracy. Adding it produced 22.95% ensemble validation but only 23.21% test accuracy, so it was excluded.

---

## 8. Five-Fold Out-of-Fold Experiment

File: [`src/train_oof_dcn.py`](src/train_oof_dcn.py)

The final 30 trials per class remained untouched. The first 270 trials per class were divided into five chronological blocks of 54 trials per class. Each fold trained on four blocks and validated on the fifth using a 0--2,000 ms DeepConvNet.

| Fold | Validation accuracy |
|---|---:|
| 1 | 14.10% |
| 2 | 18.30% |
| 3 | 17.45% |
| 4 | 13.46% |
| 5 | 14.60% |
| Combined OOF | **15.58% (1094/7020)** |

The five fold models averaged to **20.90%** on the fixed test set.

OOF did not beat the final ensemble, but the 13.46--18.30% spread exposed substantial temporal non-stationarity. A single chronological validation block has high uncertainty. Each fold model also saw less training data, and that loss outweighed variance reduction from averaging.

Reload the folds with:

```bash
python src/train_oof_dcn.py --folds 5 --epochs 100 --seed 42 --reuse
```

---

## 9. Ensemble Development

| Stage | Validation | Test |
|---|---:|---:|
| Reproduced original three-model ensemble | -- | 23.21% |
| Multi-seed five-model ensemble | 22.44% | 24.10% |
| Ensemble with 0--2,000 ms DCN | 22.82% | 24.23% |
| Ensemble with GraphEEGNet | **23.33%** | **24.36%** |

### Final model weights

| Model | Weight | Inductive bias |
|---|---:|---|
| EEGNet k=25, seed 42 | 3 | 100 ms filters and depthwise spatial processing |
| EEGNet k=15 SWA, seed 42 | 1 | Fast 60 ms dynamics and flatter SWA solution |
| EEGNet k=15 SWA, seed 123 | 2 | Independent stochastic trajectory |
| DeepConvNet, 0--2,000 ms | 2 | Hierarchical ERP features in a denoised window |
| GraphEEGNet, seed 42 | 2 | Explicit electrode-topology propagation |

```python
logits = (
    3 * eegnet_k25(x)
    + eegnet_k15_seed42(x)
    + 2 * eegnet_k15_seed123(x)
    + 2 * dcn_window(x[..., 50:551])
    + 2 * graph_eeg(x[..., 50:551])
)
prediction = logits.argmax(dim=1)
```

The gain comes from error decorrelation across temporal scales, seeds, input windows, and spatial priors rather than from one dominant model.

---

## 10. Final Reproduction

### Train all final models from scratch

After preparing `data/processed/eeg_dataset.npz`, the complete five-model workflow can be run with one command:

```bash
source .venv/bin/activate
python src/train_final_ensemble.py
```

The script trains all required EEGNet, windowed DeepConvNet, and GraphEEGNet checkpoints, saves them under `models/checkpoints/`, and automatically evaluates the fixed ensemble. Full training can take tens of minutes depending on hardware.

For a short end-to-end smoke test:

```bash
python src/train_final_ensemble.py --quick 2
```

To keep and reuse checkpoints that already exist:

```bash
python src/train_final_ensemble.py --reuse
```

Complete retraining reproduces the method rather than guaranteeing bit-identical weights. Stochastic optimization, CUDA kernels, and early-stopping trajectories can change the final accuracy. The fixed saved checkpoints are required for exact reproduction of the reported 24.36% result.

### Evaluate existing final checkpoints

```bash
source .venv/bin/activate
python src/evaluate_deep_ensemble.py
```

Expected output:

```text
Models: eegnet_k25_seed42=3, eegnet_k15_swa_seed42=1,
        eegnet_k15_swa_seed123=2, dcn_window_0_2000=2,
        graph_eeg_seed42=2
Validation accuracy: 23.33%
Test accuracy: 24.36%
```

Required checkpoints:

```text
models/checkpoints/best_eegnet_k25_seed42.pth
models/checkpoints/eegnet_k15_swa_seed42_standalone.pth
models/checkpoints/eegnet_k15_swa_seed123.pth
models/checkpoints/dcn_window_0_2000_seed42.pth
models/checkpoints/graph_eeg_seed42.pth
```

The command was executed three consecutive times with identical output. Checkpoint-based inference is deterministic in the tested environment. Full retraining is more sensitive to GPU kernels, software versions, early stopping, and random-number consumption order.

---

## 11. Main Scientific Findings

1. **Time localization matters.** Models retaining explicit ERP time bins outperformed global-pooling networks.
2. **ERP structure matters more than stationary band power.** ShallowConvNet and the fixed filter bank were weak.
3. **The final second is mostly detrimental.** Cropping to 0--2,000 ms improved DeepConvNet.
4. **Electrode topology is useful as an ensemble prior.** GraphEEGNet improved the ensemble despite lower standalone accuracy.
5. **More capacity is not automatically better.** The Conformer memorized but failed to generalize.
6. **Temporal non-stationarity is a major limitation.** OOF folds differed by almost five percentage points.

---

## 12. Limitations

1. The dataset contains only one participant.
2. Trials are temporally ordered and class-blocked, so drift can interact with class identity.
3. Validation and test contain only 780 trials each; one trial changes accuracy by about 0.128 percentage points.
4. Repeated studies on one split create indirect test-set familiarity.
5. Graph coordinates are approximate rather than participant-specific digitized positions.
6. The 24.36% result is checkpoint-reproducible, but a full multi-run retraining distribution has not yet been measured.

---

## 13. Recommended Future Work

1. Train GraphEEGNet and EEGNet under the same OOF protocol.
2. Generate OOF logits for every architecture and fit a leakage-free low-dimensional stacker.
3. Use digitized participant-specific electrode coordinates.
4. Add session/domain-adversarial objectives to reduce chronological drift.
5. Evaluate on additional participants.
6. Reserve a new untouched test split before further ensemble searches.
7. Report means, standard deviations, and confidence intervals over multiple complete retraining runs.

The central lesson is that stronger performance came from combining appropriate EEG priors—short temporal filters, explicit ERP windows, multiple stochastic solutions, and electrode topology—rather than simply increasing depth or parameter count.
