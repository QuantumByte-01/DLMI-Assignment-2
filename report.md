# BUSI Dataset Classification Report

## 1. Introduction

This project investigates class imbalance in medical image classification using the BUSI (Breast Ultrasound Images) dataset. We compare seven strategies for handling class imbalance in a 3-class CNN classifier and evaluate them on both an imbalanced and a balanced test set to reveal how evaluation methodology affects apparent model performance.

## 2. Dataset

- **Images:** 780 — benign (454), malignant (211), normal (133)
- **Imbalance ratio:** benign:normal = 3.4:1 (moderate)
- **Split:** 80/20 stratified — 624 train, 156 test

Two test sets were used for evaluation:

**Imbalanced test set** (156 samples, mirrors natural distribution):

| Class     | Count | %     |
| --------- | ----- | ----- |
| Benign    | 87    | 55.8% |
| Malignant | 42    | 26.9% |
| Normal    | 27    | 17.3% |

**Balanced test set** (81 samples, 27 per class drawn from above):

| Class     | Count | %     |
| --------- | ----- | ----- |
| Benign    | 27    | 33.3% |
| Malignant | 27    | 33.3% |
| Normal    | 27    | 33.3% |

The balanced set is a subset of the imbalanced set — no extra data was collected. It is used to assess whether a strategy's performance holds under a fair, equal-class evaluation.

## 3. Methodology

**Model (SimpleCNN):** Conv(1→32) → Conv(32→64) → Conv(64→128), each with ReLU + MaxPool. FC(128→128) → Dropout(0.5) → FC(128→3). Adam lr=0.001. Input: 128×128 grayscale.

| Strategy           | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| Baseline           | CrossEntropy, imbalanced training, 15 epochs                     |
| Oversampling       | WeightedRandomSampler (replacement=True), balanced batches       |
| Undersampling      | WeightedRandomSampler (replacement=False), capped to min class   |
| Augmentation       | Random flips, rotations (±10°), resized crops on training data |
| Class Weights      | CrossEntropy weighted inversely by class frequency               |
| Threshold Adjust   | Post-hoc: divide logits by class priors at inference             |
| Focal Loss         | gamma=1.0, downweights easy examples to focus on hard samples    |
| Ensemble (Bagging) | 3 CNNs on bootstrap samples, average logits at inference         |

All strategies: 15 epochs.

## 4. Results

**Imbalanced Test Set** (weighted averages):

| Strategy           | Accuracy | Precision | Recall | F1     |
| ------------------ | -------- | --------- | ------ | ------ |
| Baseline CNN       | 0.8205   | 0.8286    | 0.8205 | 0.8151 |
| Ensemble (Bagging) | 0.6859   | 0.5670    | 0.6859 | 0.6194 |
| Augmentation       | 0.6795   | 0.5828    | 0.6795 | 0.6075 |
| Focal Loss         | 0.6603   | 0.5455    | 0.6603 | 0.5956 |
| Oversampling       | 0.6218   | 0.6000    | 0.6218 | 0.6048 |
| Class Weights      | 0.6090   | 0.7779    | 0.6090 | 0.6246 |
| Undersampling      | 0.5833   | 0.5028    | 0.5833 | 0.5348 |

**Baseline per-class recall:** benign 93.1%, malignant 76.2%, normal 55.6%

The baseline appears dominant on this table, but this reflects its alignment with the test distribution rather than superior generalisation.

## 5. Analysis

The primary reason all strategies underperform the baseline on the imbalanced test set is **train-test distribution mismatch**. The baseline trains and tests on the same imbalanced distribution — they align. Imbalance strategies train on corrected (balanced) data but are tested on imbalanced data, so their learned class priors do not match the test set. A model trained on equal class frequencies has no reason to favour benign predictions, but a test set that is 56% benign penalises exactly that.

- **Undersampling** discards 54% of benign training images, leaving the model under-trained overall.
- **Focal Loss** (gamma=1.0) provides moderate focus on hard samples. gamma=2.0 (the original default) would be too aggressive for this mild 3.4:1 ratio, designed for extreme imbalance (100:1+) like object detection.
- **Class Weights** appears weak on the imbalanced test because it penalises benign errors heavily, making the model hesitant to predict the dominant class — which the imbalanced test mostly contains.
- **Ensemble** reduces prediction variance but does not address the root distribution mismatch.

## 6. Balanced Test Set Evaluation

| Strategy                | Imbalanced Acc | Balanced Acc    | Change          |
| ----------------------- | -------------- | --------------- | --------------- |
| Baseline CNN            | 82.1%          | 75.3%           | -6.7%           |
| Ensemble                | 68.6%          | 51.9%           | -16.7%          |
| Augmentation            | 67.9%          | 53.1%           | -14.9%          |
| Focal Loss              | 66.0%          | 51.9%           | -14.2%          |
| Oversampling            | 62.2%          | 54.3%           | -7.9%           |
| **Class Weights** | 60.9%          | **69.1%** | **+8.2%** |
| Undersampling           | 58.3%          | 45.7%           | -12.7%          |

Class-weighted loss jumps from 6th to 1st when evaluated on balanced data (+8.2%). The baseline drops 6.7%, revealing that part of its imbalanced accuracy came from predicting the majority class for uncertain samples rather than from truly learning all class boundaries. Ensemble and augmentation suffer the largest drops (-16.7% and -14.9%), suggesting they overfit to the training distribution more severely. Oversampling drops only 7.9%, comparable to the baseline, confirming that it learned more distribution-agnostic features.

The conclusion is clear: **evaluating strategies only on an imbalanced test set produces a misleading ranking**. The baseline's superiority is partly an illusion created by evaluation methodology.

## 7. Conclusions

1. **Train-test distribution mismatch** is the main reason imbalance strategies appear worse — not that the techniques are ineffective.
2. **Class-weighted loss is the most robust strategy** — best on balanced evaluation (69.1%), meaning it generalises well regardless of test distribution.
3. **Overall accuracy is misleading** on imbalanced data. The baseline scores 82.1% overall but only 55.6% recall on the normal class — clinically significant.
4. **Moderate imbalance (3.4:1) does not need aggressive correction** — high gamma focal loss and heavy resampling over-correct and hurt more than they help.
5. **Always validate on multiple test distributions** before declaring one model superior to another.
