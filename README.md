# Longitudinal Generalization of Wearable-Based Menstrual Hormone Estimation Across a Two-Year Interval

This repository contains the analysis code accompanying the study:

**Longitudinal Generalization of Wearable-Based Menstrual Hormone Estimation Across a Two-Year Interval**

Author: **Negin MELEK**  
Department of Electrical and Electronics Engineering, Faculty of Engineering and Natural Sciences, Gümüşhane University, Gümüşhane, Türkiye

## Overview

This study evaluates whether wearable-to-hormone relationships learned during the first mcPHASES monitoring period remain predictive approximately two years later.

The primary outcomes are urinary luteinizing hormone (LH) and estrone-3-glucuronide (E3G). The confirmatory analysis uses lagged heart-rate and step-count features derived exclusively from the seven completed days preceding each hormone measurement.

The repository includes the frozen confirmatory Ridge regression analysis and the post hoc robustness analyses using Elastic Net, RBF-SVR, and Random Forest.

## Code

- `mcphases_frozen_analysis.py`  
  Frozen confirmatory analysis, including preprocessing, participant-grouped cross-validation, Ridge regression, participant-balanced evaluation, bootstrap confidence intervals, and prespecified secondary analyses.

- `mcphases_posthoc_robustness_checked.py`  
  Post hoc robustness analysis using Elastic Net, RBF-SVR, and Random Forest under the same eligibility rules, preprocessing principles, participant weighting, and Round 1 to Round 2 separation.

## Data availability

The study uses the **mcPHASES** dataset available through PhysioNet:

**mcPHASES: A Dataset of Physiological, Hormonal, and Self-Reported Events and Symptoms for Menstrual Health Tracking With Wearables, version 1.0.0**  
DOI: `10.13026/zx6a-2c81`

The mcPHASES dataset is restricted-access. Users must obtain access directly through PhysioNet and comply with the applicable data use agreement.

**No participant-level mcPHASES data are included or redistributed in this repository.**

## Requirements

Python 3.10 or later.

Main dependencies:

```text
duckdb
numpy
pandas
scipy
scikit-learn
