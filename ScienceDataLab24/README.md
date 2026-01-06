# ScienceDataLab 2024 | Crop Classification from Sentinel-2 Satellite Data

Hackathon: ScienceDataLab 2024<br>
Team: 0try<br>
Participants: 3<br>
Place: 3rd

This repository contains the 3rd-place winning solution for a hackathon task focused on classifying 6 crop types using tabular data derived from Sentinel-2 satellite imagery. The solution leverages feature engineering and a stacked ensemble model to achieve high classification accuracy.

## Task Overview

The task ivolved developing a machine learning model with highest possible accuracy of classifying agricultural crops based on time-series spectral data from Sentinel-2 satellite channels. The dataset included 9 spectral bands and a pre-calculated NDVI index across 26 time points (days from the beginning of the year, 121-296).

## Dataset Description

The dataset included four distinct splits:
- Train: CSV files for training/validation
- Open-set: Intermediate evaluation test CSV files during the hackathon
- Closed-set: Final evaluation test CSV files (score is announced at the awards)

Each dataset contains 10 CSV files:
- 9 files named `B**.csv` representing different spectral channels (B02, B03, B04, B05, B06, B07, B8A, B11, B12)
- 1 file named `NDVI.csv` containing pre-calculated NDVI values

Each file contains 26 numerical columns representing spectral values at different days from the start of the year. The training data additionally includes a `culture` column with crop type labels.

## Methodology

### Feature Engineering
We created multiple vegetation indices and combined features to capture complex relationships in the spectral data:

**Core Indices:**
- NDVI (Normalized Difference Vegetation Index)
- NDWI (Normalized Difference Water Index)
- NDRE (Normalized Difference Red Edge Index)
- MSAVI (Modified Soil Adjusted Vegetation Index)
- RECI (Red Edge Chlorophyll Index)

**Combined Features:**
- NDVWI (NDVI × NDWI)
- NDSAVI (NDVI × MSAVI)
- NDVIRE (NDVI × NDRE)
- NDAWI (2NDVI × 2MSAVI × 3NDWI / 6)
- 13 custom features (TEST1-TEST13) derived from mathematical combinations of core and combined indices with weighted coefficients and biases

### Model Architecture
We implemented a stacked ensemble classifier with the following structure:

**Preprocessing Pipeline:**
1. `SimpleImputer` - Fills missing values with column means
2. `StandardScaler` - Standardizes features to zero mean and unit variance

**Stacking Classifier:**
- Base estimators (wrapped in `CalibratedClassifierCV`):
  - Logistic Regression
  - Random Forest Classifier
  - K-Neighbors Classifier
  - NuSVC
  - LinearSVC (default estimator in `CalibratedClassifierCV`)
  - ElasticNetCV (can't be used in `CalibratedClassifierCV`)
- Final estimator: `MLPClassifier` with increased hidden layers (800, 600, 200, 100, 100, 50, 50 neurons)

### Training Strategy
- The model was trained on the 70%* of training dataset
- Feature importance analysis guided the selection of combined features
- Hyperparameters were optimized empirically by comparing metrics with different configurations

## Results

- **Open-set accuracy:** 99.1%
- **Closed-set accuracy:** 87.6%

### Competition Scoring:

- Open/Closed set weighting: 10%/90%
- Additional points for team's report quality: 5.5/6
- Additional points for non-standard approach: 2.5/4

Open-set comparison with approaches:
- Baseline, RandomForestClassifier alone: ~0.966
- MLPClassifier alone: ~0.976
- Removing the Normalizer from preprocessing: ~0.986
- StackingClassifier: ~0.99

## Code Structure

The solution is implemented in a single Jupyter notebook (`0try.ipynb`) with the following sections:

1. **Dependencies Installation** - Installs required packages
2. **Data Loading** - Reads and preprocesses training and test data
3. **Feature Engineering** - Calculates vegetation indices and creates combined features
4. **Model Pipeline** - Defines the preprocessing and stacking classifier pipeline
5. **Training** - Trains the model on the complete training dataset
6. **Prediction** - Generates predictions for test data and saves to CSV

## Key libraries

- scikit-learn (ensemble methods, preprocessing, metrics)
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (visualization)
- pickle (model serialization)

## Afterword

Thanks to organizers, CSV files with labels for the open and closed sets were provided after the hackathon. This helped us learn PyTorch and TensorFlow using this example.

*Since this was our first experience in ML, we made the mistake of leaving the train/test split in the actual submitted solution.
