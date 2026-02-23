# Report on Improved Model

## Introduction

This personal report documents the improvements made to a crop classification solution originally developed for the ScienceDataLab 2024 hackathon, as part of training in working with JAX and Flax/NNX. The original solution, which secured 3rd place, employed extensive feature engineering and a stacked ensemble of scikit‑learn models. The improved version replaces the entire pipeline with a single deep neural network built with Flax/NNX and JAX, introducing a learnable imputation layer for handling missing values and a combination of loss functions. The new model achieves significantly higher accuracy on both open‑set and closed‑set test data.

## Original Approach (Baseline)

The original solution (see `0try.ipynb` and the accompanying README) consisted of:

- **Feature Engineering**: Creation of multiple vegetation indices (NDWI, NDRE, MSAVI, RECI) and 13 handcrafted features (TEST1–TEST13) from 26 time steps of 9 spectral bands and NDVI.
- **Preprocessing**: Simple imputation (mean) followed by standard scaling.
- **Model**: A stacking classifier with base estimators (Logistic Regression, Random Forest, K‑Neighbors, NuSVC, LinearSVC, ElasticNetCV) and a final `MLPClassifier` with a deep architecture (800, 600, 200, 100, 100, 50, 50 neurons).
- **Training**: Trained on the 70% of training dataset (30% validation split).
- **Results**:
  - Open‑set accuracy: **99.1%**
  - Closed‑set accuracy: **87.6%**

The large gap between open‑set and closed‑set performance suggested overfitting or insufficient generalisation.

## Improved Approach (Deep Learning with Learnable Imputation)

The improved version (see `Untitled0.ipynb`) takes a radically different path:

### 1. **Model Architecture**
- **Input**: 260 features (concatenated values from 9 bands + NDVI across 26 time points).
- **Learnable Imputation Layer**: Replaces missing values (NaNs) with trainable parameters, one per feature. This allows the model to learn optimal imputation values during training instead of relying on a static mean.
- **Main Blocks**: Three blocks of `BatchNorm → Linear → LeakyReLU → Dropout(0.1)`. Dimensions: 260 → 512 → 256 → 512.
- **Residual Block**: A `Linear → LeakyReLU → BatchNorm → + residual` block with a skip connection to improve gradient flow.
- **Output Layer**: Linear projection to 6 classes (no softmax inside the model; the loss uses logits).

### 2. **Loss Function**
The model is trained with a **combined loss**:
```
loss = (MSE + softmax_cross_entropy) / 2
```
This hybrid loss encourages both accurate probability estimates and correct class predictions.

### 3. **Optimization**
- Optimizer: AdamW with learning rate 0.0025.
- Batch size: full dataset (7261 samples).
- Number of epochs: 310.

### 4. **Handling of Missing Data**
The original dataset contained `NaN` values (visible in the printed sample within Jupyter notebook). The learnable imputation layer (`LearnableImputeLayer`) replaces each `NaN` with a trainable scalar, allowing the model to decide the best fill values during training. This is the most significant innovation within this solution.

### 5. **Training and Evaluation**
- **Training set**: The same 7261 samples used in the original solution.
- **Evaluation**: The model was tested on the provided open‑set (1619 samples) and closed‑set (1620 samples) without any additional feature engineering.
- **Metrics**: Full classification reports with precision, recall, and F1‑score per class.

## Results

The improved model showed the following results on both test sets:

| Metric          | Open‑set (original) | Open‑set (improved) | Closed‑set (original) | Closed‑set (improved) |
|-----------------|---------------------|----------------------|------------------------|-----------------------|
| Accuracy        | 99.1%               | **99.5%**            | 87.6%                  | **99.7%**             |
| Macro avg F1    | -                   | 99.5%                | -                      | 99.7%                 |
| Weighted avg F1 | -                   | 99.5%                | -                      | 99.7%                 |

The closed‑set accuracy jumped from 87.6% to 99.7%, nearly eliminating the previous generalisation gap. Precision and recall are uniformly high across all six crop classes, with the lowest F1‑score being 98.9% (soy on open‑set) and 99.4% (fallow on closed‑set).

## Full Classification Report

### Open-set

||precision|recall|f1-score|support|
|--|--|--|--|--|
|залежь|0.98291|1.00000|0.99138|345|
|зерновые|1.00000|0.99150|0.99573|353|
|кукуруза|1.00000|1.00000|1.00000|332|
|многолетние травы|1.00000|1.00000|1.00000|190|
|овощи|0.99194|1.00000|0.99595|123|
|соя|0.99632|0.98188|0.98905|276|
|&nbsp;|
|accuracy|||0.99506|1619|
|macro avg|0.99519|0.99556|0.99535|1619|
|weighted avg|0.99512|0.99506|0.99506|1619|

### Closed-set

||precision|recall|f1-score|support|
|--|--|--|--|--|
|залежь|0.98799|1.00000|0.99396|329|
|зерновые|1.00000|0.99427|0.99713|349|
|кукуруза|0.99720|1.00000|0.99860|356|
|многолетние травы|1.00000|1.00000|1.00000|184|
|овощи|1.00000|1.00000|1.00000|102|
|соя|1.00000|0.99000|0.99497|300|
|&nbsp;|
|accuracy|||0.99691|1620|
|macro avg|0.99753|0.99738|0.99744|1620|
|weighted avg|0.99694|0.99691|0.99692|1620|

## Key Learnings and Insights

1. **Feature Engineering Was Excessive**  
   The original solution relied on different vegetation indices and 13 complex, manually designed indices (e.g., TEST1–TEST13). This network, by contrast, learned directly from the raw spectral data and provided NDVI (260 features total) and achieved better results.

2. **Learnable Imputation Is Significantly Effective**  
   Instead of imputing with the mean, the model learned per‑feature imputation values. This simple addition made the largest contribution to improving the performance of both datasets by allowing the model to adapt to missingness patterns in the test data.

3. **Combining Regression and Classification Losses Can Improve Generalisation**  
   Earlier it was discovered that combining different loss functions enhances results. Solution uses average of both MSE and cross‑entropy. In this case MSE adds more to general accuracy, while cross-entropy handles class disbalance. 

4. **Deep Architectures with BatchNorm and Residual Connections**  
   The position of batch normalisation layers highly matters (before, between and after linear layer and activation). Despite residual blocks are intended to be stacked, in this case more than 1 leads to accuracy degradation.

## Conclusion


The improved crop classification model demonstrates that a carefully designed deep neural network can outperform a complex feature‑engineering and stacking ensemble on tabular satellite data. The project served good enough as part of training in working with JAX and Flax/NNX.
