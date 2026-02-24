# ML-Stress-Test
## A Modular Robustness Evaluation Framework for Machine Learning Systems

Machine learning evaluation traditionally focuses on static performance metrics such as Accuracy, F1-score, RMSE, or R². However, production failures rarely arise from low baseline accuracy — they arise from instability under changing conditions.

ML-Stress-Test is a structured robustness evaluation framework that measures how models behave under controlled stress scenarios such as noise, missing data, feature dependency shifts, and distribution drift.

It shifts evaluation from:

“How accurate is the model?”

to:

“How stable is the model under real-world stress?”

---

# Problem Statement

In real-world ML systems:

- Sensors become noisy  
- Upstream pipelines introduce missing values  
- Feature contracts change  
- Data distributions drift  
- Class imbalance shifts  

Models that perform well on validation data can degrade unpredictably when these conditions change.

There is limited tooling that systematically evaluates degradation behavior across multiple stress dimensions in a modular and reproducible manner.

ML-Stress-Test addresses this gap.

---

# Project Goals

- Provide a modular stress-testing framework
- Support both regression and classification tasks
- Measure degradation patterns under controlled perturbations
- Compute interpretable robustness indicators
- Provide interactive evaluation through a web interface
- Enable extension with new stress types

---

# System Architecture

The system is organized into modular components:

```
ML-STRESS-TEST/
│
├── models/
│   ├── baseline.py
│   └── loader.py
│
├── stress/
│   ├── tests/
│   │   ├── noise.py
│   │   ├── missingness.py
│   │   ├── feature_drop.py
│   │   ├── covariate_shift.py
│   │   └── imbalanced_shift.py (planned)
│   │
│   ├── metrics.py
│   ├── runner.py
│   ├── report.py
│   └── schemas.py
│
├── templates/
├── static/
├── app.py
└── README.md
```

Key architectural principles:

- Separation of concerns
- Deterministic stress simulation via seed control
- Safe metric computation wrappers
- Task-aware evaluation (automatic classification/regression detection)
- Extensibility for additional stress tests

---

# Web Interface

The project includes a Flask-based web application that provides:

- Baseline model evaluation
- Selectable stress test execution
- Degradation tables and summaries
- Stability classification
- Robustness scoring overview

The web layer transforms raw metric degradation into an interpretable evaluation dashboard, making the framework usable beyond CLI experimentation.

---

# Implemented Stress Tests

## Baseline Evaluation Engine

- Automatic task detection
- Multi-metric support
- Safe metric registry
- Target NaN handling
- Structured reporting

---

## Noise Injection Stress Test

Simulates Gaussian noise scaled by feature standard deviation.

Measures:

- Metric degradation across noise levels
- Absolute and percentage drop
- Noise robustness summary

Purpose:
Evaluate sensitivity to feature instability and sensor noise.

---

## Feature Drop Sensitivity Test

Simulates removal of top-k important features.

Measures:

- Performance degradation after feature removal
- Feature fragility index
- Sensitivity ranking

Purpose:
Identify over-reliance on specific features and data contract risks.

---

## Missingness Shock Test

Injects controlled missing values (5%, 10%, 20%, 40%).

Measures:

- Degradation curve
- Missingness tolerance threshold
- Stability classification

Purpose:
Evaluate resilience to upstream pipeline degradation.

---

## Covariate Shift Simulation

Simulates distribution drift via:

- Feature scaling perturbation
- Mean shifting
- Range modification
- Category dropout
- Category substitution

Measures:

- Degradation percentage
- Shift Sensitivity Index (area-under-degradation curve)
- Stability classification (stable / moderate / fragile)

Purpose:
Evaluate robustness to distribution shift — a primary cause of production ML failure.

---

# Planned Extensions

## Class Imbalance Shift (Classification)

- Controlled imbalance alteration
- Recall degradation tracking
- Balanced accuracy monitoring
- Metric illusion detection

## Robustness Scoring Engine

- Normalized degradation aggregation
- Early-collapse penalty
- Area-under-curve scoring
- Overall robustness index

## Reporting & Export Enhancements

- JSON export
- CSV export
- Config-driven experiment runs
- Structured experiment metadata logging

---

# Engineering Considerations

- Modular stress test design
- Configurable experiment parameters
- Reproducible results via fixed seeds
- Safe handling of metric computation signatures
- Task-aware evaluation logic
- No over-engineering; extensible but controlled complexity

---

# Why This Matters

Most ML repositories demonstrate:

- Model training
- Hyperparameter tuning
- Validation accuracy

Few demonstrate:

- Structured failure analysis
- Degradation modeling
- Robustness quantification
- Stability classification

ML-Stress-Test formalizes robustness evaluation as a first-class engineering process.

---

# Running the Web App

```
python app.py
```

Then navigate to:

```
http://127.0.0.1:5000
```

---

# Summary

ML-Stress-Test is a modular robustness evaluation framework designed to analyze how machine learning systems behave under stress.

It moves evaluation beyond static metrics and provides structured insights into stability, fragility, and degradation behavior — critical for real-world ML deployment.
