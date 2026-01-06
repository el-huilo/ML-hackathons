# ScienceDataLab 2025 | Semantic Segmentation of Mining Sites from Sentinel-2 Imagery

Hackathon: ScienceDataLab 2025<br>
Team: 0try<br>
Participants: 2<br>
Place: 7th

This repository contains our team's solution for the ScienceDataLab 2025 hackathon, focused on semantic segmentation of mining sites using multi-spectral satelite imagery from Sentinel-2. The solution implements a two-level ensemble approach combining deep learning segmentation models with classifiers.

## Task Overview

The task involved developing a model to detect areas and classify three types of mining-related objects in satelite imagery:
- Waste Dumps: Embankments of waste rock
- Quarry Pits: Excavation pits
- Tailings Ponds: Mineral processing waste storage facilities

The challenge required semantic segmentation of these objects in 256×256 pixel tiles with 6 spectral bands from Sentinel-2 satellite.

## Dataset Description

The dataset included four distinct splits:
- Train: Images with corresponding segmentation masks
- Validation: Images with masks for model evaluation during training
- Open-set: Intermediate evaluation test images during the hackathon
- Closed-set: Final evaluation test images (score is announced at the awards)

Each image contains 6 spectral bands:
- B02: Blue
- B03: Green
- B04: Red
- B08: NIR
- B11: SWIR 1
- B12: SWIR 2

All data was provided in TIFF format.

## Methodology

### Preprocessing and Augmentation
- Images were normalized with parameters mean=-1, std=2 because of feature or bug in T.Normalize
- Data augmentation included random horizontal/vertical flips (50% probability) and elastic transformations to alter samples across epochs
- Class imbalance was addressed through class weighting in the Random Forest classifier

### Model Architecture
We implemented a two-level ensemble with the following structure:

First Level - Segmentation Models:

- U-Net with mit_b5 encoder pre-trained on ImageNet
- FPN (Feature Pyramid Network) with mit_b5 encoder pre-trained on ImageNet

Both models were trained simultaneously, with loss calculated as the average of their predictions.

Second Level - Random Forest Classifier:
- Takes as input: 6 original spectral channels + 2 segmentation masks from first-level models
- Performs per-pixel classification with class weighting based on validation set distribution

### Training Strategy
1. Segmentation Model Training:
- Joint training of U-Net and FPN models
- Loss: Cross-entropy between ground truth and average of model predictions
- Metric: mIoU (mean Intersection over Union) calculated for the target 3 classes (excluding background)
- Early stopping based on validation loss improvement
- Best model weights saved based on validation accuracy

2. Random Forest Training:
- Trained on validation set (or small part of train set in actual submitted solution) predictions from segmentation models
- Input features: Flattened pixel values across all channels
- Output: Per-pixel class predictions

### Technical Implementation
- Batch Processing: Batch size of 11 optimized (almost limit of available memory)
- Feature Engineering: Initial experiments with vegetation indices (NDVI, MNDWI, etc.) showed no improvement and were discarded
- Tensor Processing: Complex data pipeline for converting 4D tensors to 2D feature arrays for Random Forest

## Results

- **Open-set accuracy:** 98.86%
- **Closed-set accuracy:** 77-78%

### Competition Scoring:

- Open/Closed set weighting: 20%/80% (changed from previous year's 10%/90%)
- Additional points for team's report quality: 5.5/6
- Additional points for non-standard approach: 3/4

## Code Structure

The solution is implemented in a single Jupyter notebook (`0try.ipynb`) with the following sections:

1. **Dependencies Installation** - Installs required packages
2. **Dataset Class** - Custom PyTorch Dataset for loading and augmenting TIFF images
3. **Data Preparation** - Loading train/validation/test datasets
4. **Model Definition** - U-Net and FPN model initialization
5. **Training Functions** - Custom training loop with early stopping
6. **Random Forest Integration** - Feature extraction and classifier training
7. **Inference Pipeline** - Generating predictions for test sets
8. **Results Export** - Saving segmentation masks as TIFF files

## Key libraries

- PyTorch: Deep learning framework
- segmentation-models-pytorch: Pre-trained segmentation models
- scikit-learn: Random Forest classifier
- rasterio: Geospatial raster data processing
- torchvision: Image transformations and augmentations
- matplotlib: Image visualization

## Afterword

The labeled parts of the test datasets (answers) had not been provided after the hackathon, limiting our solution improvement and investigation of errors that led to significant accuracy gap between test sets scores.

Special thanks to authors of the "EnsembleEdgeFusion: advancing semantic segmentation in microvascular decompression imaging with innovative ensemble techniques"
