# Binary Tropical Cyclone Maturity Classifier  
**Stacked Ensemble with AutoGluon**

## Overview

This repository implements a **binary classifier for tropical cyclone maturity**, which helps to determine the timing of cyclogenesis, built using a **stacked ensemble strategy**. The classifier operates on cluster-level features obtained from ensembles of forecast trajectories produced by applying the TempestExtremes framework to forecasted maps, which cover up to 15 days generated from GenCast. It then provides the probability that a specific cluster represents a *mature tropical cyclone*.

The final predictor is a **multi-layer ensemble** that combines multiple base binary classifiers and a higher-level **AutoGluon-Tabular stacking ensemble** trained on the probabilistic outputs of those base models.

The best-performing model corresponds to the AutoGluon ensemble trained with **`seed = 2`**, which achieved the highest AUC-PR (0.9610).

---

## Methodology Summary

### 1. Base Classifiers

Three different binary classification architectures were independently trained:
- Support Vector Machines (SVM)
- Neural Networks (NN, PyTorch-based)
- XGBoost (XGB)

For each architecture, **30 independent models** were trained using different random seeds to reduce sensitivity to stochastic variability.

From these, a **subset of nine base models** is used per stacking realization:
- 3 Neural Networks  
- 3 XGBoost models  
- 3 SVM models  

Each base model outputs a class probability \( p(y=1) \) for a given trajectory cluster. These probabilities are concatenated into a stacking feature vector of size 9 which is treated as a fixed input predictor for the meta-ensemble.

---

### 2. AutoGluon Meta-Ensemble

The stacking features are used to train **30 AutoGluon ensembles**, one per random seed, using:
- **Framework**: AutoGluon-Tabular  
- **Preset**: `best_quality`  
- **Optimization metric**: Average Precision (AUC-PR)  
- **Time budget**: 10 minutes per run  

AutoGluon automatically trains and combines multiple model families (ExtraTrees, CatBoost, NeuralNetTorch, etc). The final predictor is a weighted ensemble at the last stacking level, trained using a small subset of the most contributive models selected according to validation AUC-PR.
No dataset-specific hyperparameter optimization is performed. Hyperparameters are drawn from AutoGluon’s **`zeroshot` portfolio**, derived from large-scale offline benchmarking.

---

## Inference Pipeline

At inference time, the classifier follows these steps:

1. **Input**  
   A pandas DataFrame where each row corresponds to a trajectory cluster, described by cluster-level features (number of trajectories, spatial dispersion and hours prior to the cyclone happening).

2. **Base-model inference**  
   The input features are passed through the selected NN, XGB, and SVM base models to generate **nine probability values**.

3. **Stacking**  
   These probabilities form a new DataFrame that serves as input to the AutoGluon predictor.

4. **Final output**  
   Probability of being a mature tropical cyclone  


### document generated automatically, but edited manually, with chatGPT. 2026.01.19