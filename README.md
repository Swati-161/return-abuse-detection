# Return Abuse Detection and Risk Scoring

An end-to-end machine learning pipeline for detecting suspicious return and coupon-abuse behavior and assigning customers a risk score.

The project combines unsupervised anomaly detection with supervised classification. It also includes model calibration, SHAP-based explanations, and a simple policy simulator to study the effect of different intervention thresholds.

## Overview

The pipeline uses the Olist e-commerce dataset and generates customer-level behavioral features. Synthetic return-abuse behavior is then introduced to create different types of customer profiles.

The main customer archetypes are:

* Normal
* Impulse
* Serial
* Fraud

The system uses both behavioral patterns and supervised learning to identify customers who are more likely to abuse return/coupon policies.

## Pipeline

The project is divided into multiple phases:

```text
Olist Dataset
     ↓
Data Processing
     ↓
Synthetic Abuse Simulation
     ↓
Feature Engineering
     ↓
Anomaly Detection
(Isolation Forest + Autoencoder)
     ↓
Supervised Model
(XGBoost)
     ↓
Probability Calibration
(Platt Scaling)
     ↓
Risk Score (0–100)
     ↓
SHAP Explanations
     ↓
Policy Simulation
```

## Models

### 1. Isolation Forest

Used to identify customers whose behavior is unusual compared with the rest of the dataset.

### 2. Autoencoder

A neural network is trained to reconstruct normal customer behavior. Higher reconstruction error indicates more anomalous behavior.

### 3. XGBoost

A supervised XGBoost classifier is used to predict the probability of return/coupon abuse.

The training pipeline uses class weighting because the abuse class is much smaller than the normal class.

### 4. Probability Calibration

The raw XGBoost probabilities are calibrated using Platt scaling so that the resulting probabilities are more meaningful for risk-based decisions.

The calibrated probability is converted into a risk score from 0 to 100:

```text
0–30   → Safe
31–70  → Monitor
71–100 → High Risk
```

## Explainability

SHAP is used to understand why the model assigns a high or low risk score to an individual customer.

This helps identify the behavioral features contributing most to the prediction instead of treating the model as a black box.

## Policy Simulation

The project also includes a simple policy simulator to study how different risk thresholds affect interventions.

This can be used to compare the trade-off between:

* catching more abusive customers
* incorrectly flagging legitimate customers
* number of interventions
* estimated policy impact

## Results

The main supervised model achieved:

| Metric  |  Score |
| ------- | -----: |
| PR-AUC  | 0.6631 |
| ROC-AUC | 0.8419 |

The dataset contains **96,096 Olist customers**.

The simulated customer population was divided into four behavioral archetypes:

| Archetype | Customers |
| --------- | --------: |
| Normal    |    65,043 |
| Impulse   |    17,634 |
| Serial    |     9,451 |
| Fraud     |     3,968 |

## Project Structure

```text
return-abuse-detection/
│
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── simulate.py
│   ├── validate.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   └── models/
│       ├── autoencoder.py
│       └── train_model.py
│
├── run_phase1.py
├── run_phase2.py
├── run_phase3.py
├── run_phase4.py
├── run_phase6.py
│
├── requirements.txt
├── .gitignore
└── return-abuse-detection-brief.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Swati-161/return-abuse-detection.git
cd return-abuse-detection
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The project expects the Olist dataset to be available in the `data/` directory. Generated datasets, models and plots are excluded from Git using `.gitignore`.

## Running the Pipeline

The different stages can be run using the phase scripts:

```bash
python run_phase1.py
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase6.py
```

The scripts generate intermediate datasets, trained models, evaluation results and plots inside the output directories.
