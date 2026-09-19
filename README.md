# Healthcare Claims ML Pipeline

An end-to-end machine learning pipeline for exploring **health insurance claim outcome prediction** using multi-source customer, claims, and loyalty data.

The project covers the complete ML workflow: synthetic data generation, data integration, preprocessing, feature engineering, model training, evaluation, and visualisation.

Three classification approaches are compared — **Logistic Regression, Random Forest, and XGBoost** — to predict whether an insurance claim is **Approved** or **Rejected**.

---

## Overview

Insurance data is rarely contained in a single source. Customer demographics, claims history, policy information, behavioural data, and external indicators can all contribute to analytical workflows.

This project explores how these different sources can be brought together into a reproducible machine learning pipeline.

```text
Customer Data ────────┐
                      │
Claims Data ──────────┤
                      ├──► Data Integration
Loyalty / Renewal ────┤          │
                      │          ▼
External Data ────────┘     Preprocessing
                                 │
                                 ▼
                         Feature Engineering
                                 │
                                 ▼
                          Train / Test Split
                                 │
                  ┌──────────────┼──────────────┐
                  ▼              ▼              ▼
             Logistic        Random         XGBoost
             Regression      Forest
                  │              │              │
                  └──────────────┼──────────────┘
                                 ▼
                         Model Evaluation
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                 Accuracy        F1        ROC-AUC
```

---

## Problem

The machine learning task is a **binary classification problem**.

Given information about a customer and their insurance claim, the models attempt to predict:

| Outcome | Label |
| --- | ---: |
| Approved | `1` |
| Rejected | `0` |

The goal of the project is not only to train a classifier, but to demonstrate a structured pipeline for combining heterogeneous insurance data and comparing different modelling approaches.

---

## Data

The project works with multiple datasets representing different parts of an insurance environment.

### Customer Data

Synthetic customer records containing demographic and policy information such as:

- Age
- Gender
- Region
- Policy status

The generator creates approximately **8,000 customer records**.

### Claims Data

Synthetic insurance claim records containing:

- Customer ID
- Disease / medical condition
- Claim amount
- Claim outcome

Approximately **10,000 claims** are generated across medical categories including:

- Hypertension
- Diabetes
- Asthma
- Cardiac conditions
- Orthopaedic conditions
- Oncology
- Ophthalmic conditions
- ENT conditions

Claim outcomes are generated probabilistically based on factors including the medical condition and claim amount.

### Renewal & Loyalty Data

Additional behavioural features are generated for each customer, including:

- Customer tenure
- Loyalty points
- Previous claims
- Renewal status

These variables simulate longer-term customer engagement with the insurer.

### External Data

The pipeline also supports the integration of external complaint information to explore whether regional insurance complaint patterns can provide additional context.

---

## Synthetic Data Generation

Rather than relying entirely on a static dataset, the project contains dedicated data generators for customer, claims, and loyalty information.

The synthetic claims generator introduces relationships between variables instead of assigning claim outcomes completely at random.

For example:

```text
Medical Condition ──┐
                    ├──► Claim Approval Probability
Claim Amount ───────┘
```

Higher claim amounts and selected medical categories influence the probability of approval.

The loyalty dataset similarly introduces relationships between customer tenure, engagement, previous claims, loyalty points, and renewal behaviour.

This allows the complete pipeline to be reproduced without exposing real customer or healthcare information.

---

## Data Processing

The preprocessing pipeline performs the steps required to transform the merged datasets into features suitable for machine learning.

### Numerical Features

Numerical variables are transformed using:

`StandardScaler`

### Categorical Features

Categorical variables are transformed using:

`OneHotEncoder(handle_unknown="ignore")`

### Pipeline Design

Scikit-learn's `ColumnTransformer` is used to apply the appropriate transformation to each feature type.

The preprocessing stage is then combined with each classifier using a Scikit-learn `Pipeline`.

This keeps preprocessing and model training together and helps prevent inconsistencies between training and inference.

---

## Models

Three classification algorithms are evaluated.

### Logistic Regression

Used as the baseline linear classifier.

It provides a useful reference point for determining whether more complex models provide meaningful improvements.

### Random Forest

An ensemble of decision trees capable of modelling nonlinear relationships and interactions between features.

The implementation uses **300 estimators**.

### XGBoost

Gradient-boosted decision trees are used as the third modelling approach.

The current configuration uses:

```text
Estimators       400
Maximum depth    5
Learning rate    0.05
Subsample        0.90
Column sample    0.90
```

---

## Model Evaluation

Models are compared using several metrics rather than accuracy alone.

- **Accuracy** — proportion of predictions classified correctly
- **F1 Score** — balances precision and recall
- **ROC-AUC** — measures the model's ability to distinguish approved from rejected claims across classification thresholds

The pipeline also generates:

- ROC curves
- Confusion matrices
- Model comparison tables

---

## Results

The current experiment produced the following results:

| Model | Accuracy | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: |
| **Logistic Regression** | **57.1%** | **0.681** | **0.553** |
| Random Forest | 53.4% | 0.618 | 0.526 |
| XGBoost | 53.5% | 0.628 | 0.519 |

In this experiment, **Logistic Regression produced the strongest overall results**, despite being the simplest of the three models.

The results also demonstrate an important machine learning consideration: increased model complexity does not necessarily produce better generalisation.

The relatively modest ROC-AUC values indicate that the available synthetic features provide limited separation between approved and rejected claims. The project therefore focuses on the **pipeline, experimentation process, and comparison of modelling approaches** rather than presenting the resulting classifier as a production insurance decision system.

---

## Repository Structure

```text
healthcare-claims-ml-pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── config.py
│   ├── generate_customers.py
│   ├── generate_claims.py
│   ├── generate_loyalty.py
│   ├── preprocessing.py
│   ├── modelling.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

**Language**

Python

**Data Processing**

Pandas · NumPy

**Machine Learning**

Scikit-learn · XGBoost

**Visualisation**

Matplotlib

**ML Techniques**

Binary Classification · Feature Engineering · One-Hot Encoding · Feature Scaling · Ensemble Learning · Gradient Boosting · Model Evaluation

---

## Key Engineering Concepts

This project demonstrates:

- Multi-source data integration
- Synthetic dataset generation
- Reproducible ML workflows
- Numerical and categorical preprocessing
- Scikit-learn `ColumnTransformer`
- End-to-end ML pipelines
- Binary classification
- Ensemble learning
- Gradient boosting
- Model comparison
- Confusion-matrix analysis
- ROC-AUC evaluation

---

## Limitations

This project is an experimental and educational machine learning system and **should not be used to make real insurance or healthcare decisions**.

Several important limitations apply:

- Much of the data is synthetically generated.
- Synthetic relationships do not represent real insurance underwriting or claim-assessment policies.
- Some available features may have limited predictive relationship with claim outcomes.
- The current dataset does not represent the scale, complexity, regulatory requirements, or distribution shifts found in production insurance systems.
- Model performance is not sufficient for deployment as a real claim decision system.

The results should therefore be interpreted as an exploration of **data engineering and machine learning workflow design**, rather than evidence of real-world claim prediction capability.

---

## Future Improvements

Potential extensions include:

- Feature ablation experiments to measure the contribution of each data source
- Cross-validation instead of relying on a single train/test split
- Hyperparameter optimisation
- Precision-recall analysis
- Class imbalance analysis
- Feature importance and SHAP-based model interpretation
- Improved synthetic data generation with richer relationships between variables
- Experiment tracking
- Model persistence and versioning
- REST API for model inference
- Automated testing and CI/CD

One particularly useful experiment would compare progressively richer feature sets:

```text
Claims only
    ↓
Claims + Customer Data
    ↓
Claims + Customer + Loyalty
    ↓
Claims + Customer + Loyalty + External Data
```

This would make it possible to quantify whether each additional data source provides useful predictive information.

---

## Purpose

This project was developed as part of my postgraduate work in **Artificial Intelligence and Machine Learning**.

It reflects my interest in combining my background in production software engineering with machine learning, data engineering, and AI systems.

For more of my work:

**Portfolio:** [aeman.com.au](https://www.aeman.com.au/)  
**GitHub:** [github.com/Aeman-Fatima](https://github.com/Aeman-Fatima)
