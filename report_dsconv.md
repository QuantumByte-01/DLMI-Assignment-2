# BUSI Dataset Classification Report: Depthwise Separable CNN

## 1. Introduction

This report extends the BUSI class-imbalance study by swapping SimpleCNN for a Depthwise Separable CNN (DSConvCNN). Each standard 3×3 convolution is replaced with a depthwise + pointwise pair, cutting convolutional parameters by ~8-9× per layer. The same seven imbalance strategies are evaluated on both imbalanced and balanced test sets.

## 2. Dataset

- **Images:** 780; benign (437), malignant (210), normal (133)
- **Imbalance ratio:** benign:normal ≈ 3.3:1 (moderate)
- **Split:** 80/20 stratified, 624 train, 156 test

Two test sets were used for evaluation:

**Imbalanced test set** (156 samples, mirrors natural distribution):

| Class | Count | % |
|---|---|---|
| Benign | 87 | 55.8% |
| Malignant | 42 | 26.9% |
| Normal | 27 | 17.3% |

**Balanced test set** (81 samples, 27 per class drawn from above).

## 3. Model Architecture

Standard conv costs k²·C_in·C_out parameters. Depthwise separable splits this into a depthwise conv (k²·C_in, one filter per channel) and a pointwise 1×1 conv (C_in·C_out), giving ~8-9× fewer conv params for 3×3 kernels.

**DSConvCNN:** Standard Conv2d(1→32) as entry (single-channel input has nothing to separate), then DepthwiseSeparable(32→64), DepthwiseSeparable(64→128), each with BatchNorm + ReLU + MaxPool. FC(128×16×16→128) → Dropout(0.5) → FC(128→3). Adam lr=0.001. Input: 128×128 grayscale.

| Model | Total Params |
|---|---|
| SimpleCNN | 4,287,491 |
| DSConvCNN | 4,206,659 |
| Reduction | 1.9% |

The overall reduction is small because the FC head alone accounts for 4,194,432 parameters in both models. The conv layers themselves see a much larger reduction. Replacing the FC head with Global Average Pooling would bring the total reduction closer to the per-layer 8-9× figure.

## 4. Methodology

| Strategy | Description |
|---|---|
| Baseline | CrossEntropy, imbalanced training, 15 epochs |
| Oversampling | WeightedRandomSampler (replacement=True), balanced batches |
| Undersampling | WeightedRandomSampler (replacement=False), capped to min class |
| Augmentation | Random flips, rotations (±10°), resized crops on training data |
| Class Weights | CrossEntropy weighted inversely by class frequency |
| Threshold Adjust | Post-hoc: divide logits by class priors at inference |
| Focal Loss | gamma=1.0, downweights easy examples |
| Ensemble (Bagging) | 3 DSConvCNNs on bootstrap samples (5 epochs each), average logits |

All strategies except ensemble were trained for 15 epochs.

## 5. Results

**Imbalanced Test Set** (weighted averages):

| Strategy | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Oversampling | 0.8077 | 0.8091 | 0.8077 | 0.8035 |
| Class Weights | 0.7692 | 0.7740 | 0.7692 | 0.7708 |
| Focal Loss | 0.7179 | 0.6091 | 0.7179 | 0.6484 |
| Baseline DSConvCNN | 0.7115 | 0.5886 | 0.7115 | 0.6418 |
| Ensemble (Bagging) | 0.6987 | 0.7509 | 0.6987 | 0.6453 |
| Augmentation | 0.6795 | 0.6234 | 0.6795 | 0.6009 |
| Undersampling | 0.6603 | 0.7239 | 0.6603 | 0.5975 |
| Threshold Adjust | 0.6026 | 0.7662 | 0.6026 | 0.6059 |

**Baseline per-class recall:** benign 93%, malignant 71%, normal 0%

The baseline completely fails on the normal class; every normal sample gets misclassified.

**Balanced Test Set** (27 per class):

| Strategy | Imbalanced Acc | Balanced Acc | Bal Precision | Bal Recall | Bal F1 |
|---|---|---|---|---|---|
| Oversampling | 80.8% | 75.3% | 0.7793 | 0.7531 | 0.7540 |
| Class Weights | 76.9% | **76.5%** | 0.7782 | 0.7654 | 0.7674 |
| Focal Loss | 71.8% | 58.0% | 0.4421 | 0.5802 | 0.4829 |
| Baseline DSConvCNN | 71.2% | 55.6% | 0.3902 | 0.5556 | 0.4507 |
| Ensemble (Bagging) | 69.9% | 55.6% | 0.7240 | 0.5556 | 0.5016 |
| Augmentation | 67.9% | 53.1% | 0.4718 | 0.5309 | 0.4437 |
| Undersampling | 66.0% | 50.6% | 0.7229 | 0.5062 | 0.4321 |

## 6. Analysis

Oversampling takes the top spot at 80.8% on the imbalanced test, which is different from the SimpleCNN experiments where class weights dominated. The DSConvCNN's BatchNorm layers and different learning dynamics seem to benefit more from balanced batches than from weighted loss during early training.

However, class weights drops only 0.4% from imbalanced to balanced evaluation (76.9% → 76.5%), compared to oversampling's 5.5% drop (80.8% → 75.3%). This makes class weights the more reliable choice when the deployment distribution is unknown.

The normal class is the critical failure point. Baseline, focal loss, and augmentation all hit 0% normal recall, meaning they learn to classify everything as benign or malignant. Only oversampling (59% normal recall) and class weights (70% normal recall) actually detect normal cases. In a clinical setting, failing on normal tissue means unnecessary procedures for healthy patients.

Undersampling (66.0%) struggles more than it did with SimpleCNN because throwing away ~300 benign images leaves too little data for the DSConvCNN to converge in 15 epochs. The ensemble also underperforms (69.9%) because each sub-model only trains for 5 epochs, and DSConvCNN converges slower than SimpleCNN.

**Comparison with SimpleCNN:**

| Metric | SimpleCNN | DSConvCNN |
|---|---|---|
| Best imbalanced accuracy | 81.4% (Class Weights) | 80.8% (Oversampling) |
| Best balanced accuracy | 79.0% (Class Weights) | 76.5% (Class Weights) |

DSConvCNN trails SimpleCNN slightly. The 1.9% parameter reduction is too small to matter for accuracy, and the real benefit (computational efficiency) shows up in inference speed and memory, not test accuracy on this small dataset.

## 7. Conclusions

1. **Oversampling works best for DSConvCNN on the imbalanced test** (80.8%), but **class weights is the most stable across distributions** (only -0.4% drop to balanced test).
2. **Normal class recall is the biggest problem**: baseline, focal loss, and augmentation all score 0%. Only oversampling and class weights actually learn to detect normal tissue.
3. **Depthwise separable convolutions cut conv-layer parameters by ~8-9×**, but the FC head dominates total count, limiting overall reduction to 1.9%. Global Average Pooling would fix this.
4. **Balanced evaluation is still necessary**: oversampling's 5.5% drop between test sets shows it still leans on the majority class more than class weights does.
5. **The choice of imbalance strategy matters more than the choice of architecture** for this dataset and imbalance ratio.
