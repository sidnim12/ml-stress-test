# ML-Stress-Test

A modular robustness evaluation engine for machine learning models.

Most machine learning workflows focus on performance metrics such as accuracy, F1-score, RMSE, or R². However, real-world systems fail not because of poor baseline accuracy, but because of instability under changing conditions.

ML-Stress-Test evaluates how models behave under controlled stress scenarios such as noise, missing data, feature dependency changes, and distribution drift.

It answers a more important question:

**How gracefully does the model fail?**

---

# Motivation

In production environments:

- Sensors become noisy  
- Upstream pipelines introduce missing values  
- Feature contracts change  
- Data distributions drift  
- Class imbalance shifts  

A model that performs well on clean validation data may degrade unpredictably under these conditions.

This project provides a structured framework to simulate such stress scenarios and measure degradation behavior in a controlled and reproducible way.

---

# Key Features

- Modular stress testing architecture
- Supports both regression and classification tasks
- Automatic task detection
- Safe metric computation wrapper
- Degradation tracking across stress levels
- Robustness scoring framework
- Interactive web-based reporting interface
- Config-driven experiment design
- Reproducible random seeds

---

# Web Interface

ML-Stress-Test includes a Flask-based web application that provides an interactive evaluation dashboard.

The web interface allows you to:

- Upload datasets
- Train and evaluate baseline models
- Run selected stress tests
- View degradation tables
- Inspect robustness summaries
- Review stability classifications

This makes the tool usable not only as a Python module but also as an evaluation dashboard for experimentation and analysis.

Core web components:

- `app.py` – Flask entry point  
- `templates/` – Structured HTML reports  
- `static/` – Styling and UI assets  

---

# Architecture

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

Design principles:

- Clear separation of concerns
- Modular stress test implementations
- Deterministic experiment configuration
- Extensible architecture for future stress types
- Task-aware metric handling

---

# Stress Tests Implemented

## Baseline Evaluation

- Automatic task detection (regression vs classification)
- Multi-metric computation
- Safe metric wrapper
- Target NaN handling

Provides structured baseline performance metrics.

---

## Noise Injection Stress Test

Simulates increasing Gaussian noise in numeric features.

Measures:

- Absolute metric degradation
- Percentage drop
- Noise robustness summary

Purpose:
Evaluate sensitivity to sensor noise or feature instability.

---

## Feature Drop Sensitivity Test

Simulates removal of top-k important features.

Measures:

- Performance before and after drop
- Feature fragility index
- Sensitivity ranking

Purpose:
Detect over-reliance on specific features and data contract risks.

---

## Missingness Shock Test

Injects controlled levels of missing values.

Measures:

- Degradation curve
- Missingness tolerance threshold
- Stability classification

Purpose:
Evaluate resilience to upstream data pipeline failures.

---

## Covariate Shift Simulation

Simulates distribution drift using:

- Feature scaling perturbation
- Mean shifting
- Range clipping
- Category dropout
- Category substitution

Measures:

- Degradation percentage
- Shift Sensitivity Index (AUC-based)
- Stability classification (stable / moderate / fragile)

Purpose:
Evaluate robustness to real-world distribution drift.

---

# Planned Extensions

## Class Imbalance Shift (Classification)

- Controlled imbalance alteration
- Recall degradation tracking
- Balanced accuracy behavior
- Metric illusion detection

## Robustness Scoring Engine

- Normalized degradation aggregation
- Early-collapse penalty
- Area-under-degradation scoring
- Overall robustness index

## Reporting & Export

- JSON export
- CSV export
- Config-driven experiment runs
- Structured experiment metadata logging

---

# Example Use Cases

- Pre-deployment model validation
- Risk analysis for production ML systems
- Academic research on model stability
- Teaching robustness and model evaluation concepts
- Comparing model architectures under stress

---

# Engineering Principles

- Modular and extensible design
- Reproducible experiments
- Safe metric computation
- Task-aware evaluation
- No premature optimization
- Transparent degradation tracking

---

# Installation

```bash
git clone https://github.com/your-username/ml-stress-test.git
cd ml-stress-test
pip install -r requirements.txt
```

---

# Running the Web App

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

# Contribution

Contributions are welcome.

If you would like to:

- Add new stress tests
- Improve scoring mechanisms
- Enhance reporting
- Optimize performance

Please open an issue or submit a pull request.

---

# Summary

ML-Stress-Test transforms traditional model evaluation into a structured robustness assessment framework.

Instead of focusing solely on performance metrics, it measures stability under controlled stress conditions — providing deeper insight into model reliability before deployment.
