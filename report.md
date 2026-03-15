# BUSI Dataset Classification Report

## 1. Introduction

This project investigates class imbalance in medical image classification using the BUSI (Breast Ultrasound Images) dataset. We compare seven strategies for handling class imbalance in a 3-class CNN classifier and evaluate them on both an imbalanced and a balanced test set to reveal how evaluation methodology affects apparent model performance.

## 2. Dataset

- **Images:** 780 — benign (454), malignant (211), normal (133)
- **Imbalance ratio:** benign:normal = 3.4:1 (moderate)
- **Split:** 80/20 stratified — 624 train, 156 test

Two test sets were used for evaluation:

**Imbalanced test set** (156 samples, mirrors natural distribution):

| Class | Count | % |
|---|---|---|
| Benign | 87 | 55.8% |
| Malignant | 42 | 26.9% |
| Normal | 27 | 17.3% |

**Balanced test set** (81 samples, 27 per class drawn from above):

| Class | Count | % |
|---|---|---|
| Benign | 27 | 33.3% |
| Malignant | 27 | 33.3% |
| Normal | 27 | 33.3% |

The balanced set is a subset of the imbalanced set — no extra data was collected. It is used to assess whether a strategy's performance holds under a fair, equal-class evaluation.

## 3. Methodology

**Model (SimpleCNN):** Conv(1→32) → Conv(32→64) → Conv(64→128), each with ReLU + MaxPool. FC(128→128) → Dropout(0.5) → FC(128→3). Adam lr=0.001. Input: 128×128 grayscale.

| Strategy | Description |
|---|---|
| Baseline | CrossEntropy, imbalanced training, 15 epochs |
| Oversampling | WeightedRandomSampler (replacement=True), balanced batches |
| Undersampling | WeightedRandomSampler (replacement=False), capped to min class |
| Augmentation | Random flips, rotations (±10°), resized crops on training data |
| Class Weights | CrossEntropy weighted inversely by class frequency |
| Threshold Adjust | Post-hoc: divide logits by class priors at inference |
| Focal Loss | gamma=1.0, downweights easy examples to focus on hard samples |
| Ensemble (Bagging) | 3 CNNs on bootstrap samples, average logits at inference |

## 4. Results

**Imbalanced Test Set** (weighted averages):

| Strategy | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Class Weights | 0.8141 | 0.8207 | 0.8141 | 0.8138 |
| Baseline CNN | 0.7692 | 0.7771 | 0.7692 | 0.7673 |
| Ensemble (Bagging) | 0.7692 | 0.7830 | 0.7692 | 0.7504 |
| Oversampling | 0.7436 | 0.7438 | 0.7436 | 0.7388 |
| Focal Loss | 0.7436 | 0.7540 | 0.7436 | 0.7410 |
| Undersampling | 0.7244 | 0.7322 | 0.7244 | 0.7140 |
| Threshold Adjust | 0.7244 | 0.7539 | 0.7244 | 0.7256 |
| Augmentation | 0.6987 | 0.7454 | 0.6987 | 0.6555 |

**Baseline per-class recall:** benign 80.5%, malignant 83.3%, normal 55.6%

## 5. Analysis

With all strategies trained equally for 15 epochs, the results tell a different story. Class-weighted loss now ranks first (81.4%), outperforming the baseline (76.9%). This confirms that when given equal training time, imbalance-aware strategies are effective. The distribution mismatch effect is reduced when strategies are no longer under-trained.

- **Class Weights** performs best — penalising minority class errors pushes the model to learn all three classes properly, and with sufficient training this translates to higher overall accuracy even on the imbalanced test.
- **Undersampling** improves significantly versus the earlier 5-epoch run, as the model has more iterations to learn from the reduced dataset.
- **Focal Loss** (gamma=1.0) performs competitively at 74.4%, comparable to oversampling. The moderate gamma is appropriate for the 3.4:1 ratio.
- **Augmentation** ranks last — random transforms alone without class rebalancing are insufficient.
- **Baseline per-class recall** shows normal class is still the weakest (55.6%), despite good overall accuracy, illustrating why overall metrics alone are insufficient.

## 6. Balanced Test Set Evaluation

| Strategy | Imbalanced Acc | Balanced Acc | Bal Precision | Bal Recall | Bal F1 |
|---|---|---|---|---|---|
| Class Weights | 81.4% | **79.0%** | 0.8097 | 0.7901 | 0.7886 |
| Ensemble | 76.9% | 67.9% | 0.7410 | 0.6790 | 0.6735 |
| Baseline CNN | 76.9% | 74.1% | 0.7737 | 0.7407 | 0.7365 |
| Focal Loss | 74.4% | 70.4% | 0.7617 | 0.7037 | 0.7018 |
| Oversampling | 74.4% | 67.9% | 0.7276 | 0.6790 | 0.6823 |
| Undersampling | 72.4% | 69.1% | 0.7846 | 0.6914 | 0.6913 |
| Augmentation | 69.9% | 60.5% | 0.7486 | 0.6049 | 0.5401 |

Class-weighted loss maintains the top position on both test sets, dropping only 2.4% from imbalanced to balanced evaluation — the smallest drop of all strategies. The baseline drops 2.6%, while ensemble drops 9.0% and augmentation drops 9.4%. This confirms that class-weighted loss learns the most distribution-agnostic features. The balanced evaluation also exposes that strategies with decent imbalanced accuracy (ensemble at 76.9%) can still perform poorly on balanced data (67.9%), revealing hidden bias toward the majority class.

## 7. Conclusions

1. **Class-weighted loss outperforms the baseline when trained equally** — achieving 81.4% vs baseline 76.9%. However, not all imbalance strategies surpass the baseline; oversampling, undersampling, focal loss, and augmentation all score below it, showing that the choice of strategy matters significantly.
2. **Class-weighted loss is the most robust strategy** — best on both imbalanced (81.4%) and balanced (79.0%) test sets, with the smallest accuracy drop between the two.
3. **Overall accuracy is misleading** on imbalanced data. The baseline scores 76.9% overall but only 55.6% recall on the normal class — clinically significant in a medical context.
4. **Balanced test set evaluation reveals hidden bias** — strategies with similar imbalanced accuracy can differ greatly on balanced data, exposing majority-class dependence.
5. **Always validate on multiple test distributions** before declaring one model superior to another.
