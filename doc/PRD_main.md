Below is a **full project development document** (engineer-facing) in **Markdown**, derived from your manuscript up to Methods and aligned with your stated requirements: raw EHR-style inputs → configurable training mode (S/T/DR) → trained model artifacts + full evaluation outputs.  

---

# PRISM Project Development Document (Markdown)

## 1. Project Overview

### 1.1 Goal

PRISM (Predictive Renal Intelligence Survival Modeling) will build, validate, and package deep-learning survival models that estimate individualized **all-cause mortality risk** for patients with advanced CKD at the dialysis decision point. The key output is the predicted mortality risk trajectory under two treatment strategies:

* **A = 1 (early dialysis):** initiation of maintenance dialysis within **90 days after t₀**
* **A = 0 (non-early):** no dialysis initiation within 90 days after t₀

The decision index date **t₀** is defined as the first outpatient eGFR **≤ 10 mL/min/1.73m²**, while cohort screening begins at persistent eGFR < 15 mL/min/1.73m². These definitions and the causal ML framing are taken directly from the manuscript Methods section. 

### 1.2 Supported Learning Modes (Configured via YAML)

The system must support three training modes:

1. **S-learner mode:** trains **one** survival model `f(X, A)` and produces counterfactual predictions by setting A to 0 vs 1 at inference time.
2. **T-learner mode:** trains **two** survival models: `f0(X)` (A=0 group) and `f1(X)` (A=1 group).
3. **DR-learner mode:** trains a survival model with **propensity-score-based weighting** (IPTW) and produces counterfactual predictions similarly to S-learner, but under weighted training.

These approaches follow standard causal ML practice for heterogeneous treatment effect estimation in observational data. 

---

## 2. Data Contract (Input/Output)

### 2.1 Raw Input Data (Minimum Required Tables)

The pipeline assumes raw data is available in one of these formats:

* CSV/Parquet files, or
* a database extract dumped into structured files.

The **minimum required** is a patient-level longitudinal record that can be transformed into an index-date dataset at t₀.

#### Required core entities

1. **Demographics**

   * patient_id (pseudonymized)
   * sex
   * date_of_birth (or age at visit)
2. **Laboratory measurements (longitudinal)**

   * specimen_date
   * creatinine
   * bicarbonate
   * phosphate
   * hemoglobin
   * serum_albumin
   * uacr (or upcr to derive uacr)
   * optional: 24h urine protein
3. **Diagnoses / ICD-10 codes**

   * diagnosis_date
   * icd10_code
4. **Dialysis initiation data**

   * dialysis_start_date
   * modality (HD/PD if available)
   * dialysis_type (must distinguish maintenance vs temporary if possible)
5. **Mortality data**

   * death_date (if applicable)
6. **Follow-up / censoring**

   * last_contact_date (or last lab/visit date; define operationally)

### 2.2 Derived Cohort Dataset (Index-Date Level)

After cohort formation, each patient contributes one row at **t₀** with:

* covariates X at/near t₀ (using specified lookback windows)
* treatment strategy label A (early vs non-early)
* survival outcome (time-to-death or censor; event indicator)

#### Index-date feature vector X (example)

* age_at_t0
* sex
* time_since_ckd_diagnosis_days
* charlson_comorbidity_index
* bicarbonate (closest to t₀ within window)
* uacr (observed or derived)
* hemoglobin
* serum_albumin
* phosphate
* optional additional labs / derived features if enabled

### 2.3 Output Artifacts

The project must generate, depending on YAML configuration:

#### Model artifacts

* **S-learner:** one trained survival model artifact
* **T-learner:** two trained survival model artifacts
* **DR-learner:** one trained survival model + one trained propensity model + saved weight diagnostics

All model artifacts must be “fully usable”, meaning:

* loadable from disk
* usable for batch inference on new data
* versioned via MLflow
* reproducible with recorded config + code commit hash (if Git is used)

#### Evaluation artifacts

* predictive performance metrics (discrimination + calibration) on temporal and spatial test sets
* causal effect summaries (ATE/ATT at 1, 3, 5 years; and optionally full curves)
* uncertainty quantification via bootstrap
* overlap / positivity diagnostics (propensity distributions; trimming stats)
* balance metrics post-weighting (SMD pre/post)

---

## 3. Cohort Formation Logic

### 3.1 Persistent eGFR < 15 (Screening)

A patient is eligible for screening if they have persistent eGFR < 15 mL/min/1.73m² defined as:

* two outpatient eGFR values < 15
* separated by **≥ 90 days** and **≤ 365 days**
* within study period 2009–2023

### 3.2 Dialysis Decision Index Date t₀

Define **t₀** as:

* first outpatient eGFR ≤ 10 mL/min/1.73m² after screening eligibility is met

Operational rules:

* outpatient-only creatinine preferred to reduce AKI contamination
* if both outpatient and inpatient exist, outpatient takes precedence unless configured otherwise
* allow configurable smoothing (e.g., “confirmatory second eGFR ≤ 10 within 14–30 days”) as sensitivity analysis

### 3.3 Treatment Strategy Definition A

Define early dialysis as:

* dialysis_start_date ∈ [t₀, t₀ + 90 days]

Then:

* A = 1 if early dialysis
* A = 0 otherwise (including dialysis after 90 days)

### 3.4 Follow-up, Outcome, and Censoring

Primary outcome:

* **all-cause mortality** after t₀

Time-to-event:

* duration_days = min(death_date, censor_date) − t₀

Event indicator:

* event = 1 if death observed before censor
* event = 0 otherwise

Censoring date:

* last_contact_date (preferred), otherwise last recorded outpatient encounter/lab date (configurable)

---

## 4. Data Engineering & Preprocessing

### 4.1 Feature Extraction Windows

Define windows relative to t₀:

* labs: closest value within lookback window (default 90 days)
* comorbidity: Charlson computed from ICD-10 codes within lookback window (default 5 years) and/or all history (configurable)
* CKD diagnosis date: earliest CKD code date or first eGFR < 60 date (configurable approach)

### 4.2 UACR Derivation

If UACR missing, derive from UPCR using the specified published conversion approach. Implement as deterministic feature engineering in preprocessing (must be consistent across train/test).

### 4.3 Missingness and Imputation

Primary imputation:

* MICE (IterativeImputer / chained equations) for continuous variables

Rules:

* imputation is fitted on training set only
* applied unchanged to validation/test sets
* imputation model is saved with the pipeline artifact

### 4.4 Normalization and Encoding

* categorical: one-hot (sex; optionally other categorical variables)
* continuous skewed: log transform + min-max
* other continuous: standardize + min-max to [0,1]

All transformations must be implemented as a single reproducible pipeline object that can be applied at inference time.

---

## 5. Modeling: Survival + Causal Learners

### 5.1 Survival Backends

Use `pycox` for:

* **DeepSurv** (primary; single-event mortality)
* **DeepHit** (optional; if later adding competing events)

DeepSurv is the default because the manuscript’s primary endpoint is all-cause mortality. 

### 5.2 S-Learner Specification

Model form:

* train one survival model `fθ([X, A])`

Training data:

* all patients; input includes A

Inference:

* for each patient X, create two inputs:

  * `[X, A=0]` → risk curve R0(t)
  * `[X, A=1]` → risk curve R1(t)

Outputs:

* individual risk difference Δ(t) = R1(t) − R0(t)
* ATE/ATT summaries at 1/3/5 years

### 5.3 T-Learner Specification

Train:

* model `f0` on subset A=0
* model `f1` on subset A=1

Inference:

* R0(t) = f0(X)
* R1(t) = f1(X)

Notes:

* requires sufficient sample size in both arms
* higher variance than S-learner; must report uncertainty

### 5.4 DR-Learner Specification (Propensity-Weighted Survival Training)

This project implements a **doubly robust-inspired** workflow operationally as:

* train propensity model `e(X)=P(A=1|X)`
* compute stabilized IPTW weights
* train a weighted survival model (DeepSurv/DeepHit) for outcome using weights

Core components:

#### 5.4.1 Propensity model

Candidate models:

* Logistic regression (baseline)
* Gradient boosting classifier
* XGBoost/LightGBM if enabled

Output:

* e_hat for each patient

Diagnostics:

* AUROC for propensity (not the goal, but useful)
* calibration curve / Brier score
* overlap check: distributions of e_hat by A group
* positivity trimming proportion

#### 5.4.2 Stabilized weights

Let p = P(A=1) in training data.

* If A=1: w = p / e_hat
* If A=0: w = (1-p) / (1 - e_hat)

Trimming:

* default clip e_hat ∈ [0.05, 0.95] (configurable)

#### 5.4.3 Weighted survival training

Modify survival loss to accept per-sample weights.
This is the key deliverable for DR mode: the survival model must be trained using w to reduce confounding on measured covariates.

Inference:

* same as S-learner if the survival model includes A as input; otherwise DR will estimate ATE via weighted marginal predictions (configurable).
  Recommended default: **include A in the survival model** even in DR mode to support individualized counterfactual curves.

---

## 6. Hyperparameter Tuning and Training Orchestration

### 6.1 Cross-validation Scheme

The manuscript specifies **10-fold time-based cross-validation with patient grouping**. Implement this as:

* sort patients by t₀ date
* split into folds by time blocks to preserve temporal ordering
* group by patient_id (one row per patient at t₀, so grouping is trivial, but keep the interface)

### 6.2 Hyperparameter Search

Use Ray Tune or Optuna (configurable), optimizing:

* primary: integrated concordance index
* secondary: negative log-likelihood
* tertiary: integrated Brier score

Early stopping:

* based on validation loss or c-index plateau
* patience configurable

### 6.3 Experiment Tracking

All runs must log to MLflow:

* config YAML snapshot
* dataset version identifiers
* model weights
* performance metrics
* calibration plots and risk curves
* propensity diagnostics (if DR mode)

ZenML can orchestrate:

* data ingestion → preprocessing → training → evaluation → packaging

---

## 7. Evaluation Plan (Benchmarking Across S/T/DR)

### 7.1 Test Sets

Two held-out sets:

* temporal test set (index ≥ 2022-01-01)
* spatial test set (10% held out from earlier period)

No training or tuning may use these sets.

### 7.2 Predictive Metrics (Survival)

Compute at horizons 1, 3, 5 years:

* time-dependent concordance index (integrated c-index)
* Brier score + Integrated Brier Score
* calibration slope/intercept and calibration curves

For DeepHit (if used):

* CIF-based Brier score and calibration on CIF

### 7.3 Causal Effect Metrics

For each learner, compute:

* individual counterfactual risk curves R0(t), R1(t)
* individual treatment effect Δ(t) at 1/3/5 years
* ATE(t) = mean(R1(t) − R0(t))
* ATT(t) = mean(R1(t) − R0(t) | A=1)

Uncertainty:

* bootstrap 500–2000 iterations (configurable)
* report 95% percentile intervals for ATE/ATT

### 7.4 Overlap and Balance (Required for DR, Recommended for All)

Overlap diagnostics:

* e_hat distribution by A group
* percentage trimmed/clipped
* positivity warnings if large mass near 0 or 1

Balance diagnostics:

* standardized mean difference (SMD) for each covariate:

  * before weighting
  * after weighting (DR)
    Target criterion:
* SMD < 0.1 for “balanced enough” (report, do not hard-fail unless extreme)

### 7.5 Sensitivity Analyses

All sensitivity analyses must be configurable via YAML:

* dialysis “early window” W: 60/90/120 days
* alternative t₀ definition: first eGFR ≤ 12 vs ≤ 10
* propensity trimming thresholds: 0.01/0.05/0.10
* outpatient-only vs all creatinine sources for eGFR

---

## 8. System Architecture and Repository Layout

### 8.1 Suggested Repo Structure

```text
prism/
  configs/
    default.yaml
    s_learner.yaml
    t_learner.yaml
    dr_learner.yaml
  data/
    raw/               # not committed
    interim/           # cohort tables
    processed/         # model-ready datasets
  prism_core/
    ingestion/
      readers.py
      schema.py
    cohort/
      egfr.py
      t0_builder.py
      exposure.py
      outcomes.py
    features/
      charlson.py
      uacr_conversion.py
      feature_windows.py
    preprocessing/
      pipeline.py
      imputation.py
      scaling.py
    models/
      survival_pycox.py
      propensity.py
      learners/
        s_learner.py
        t_learner.py
        dr_learner.py
    training/
      tune.py
      trainer.py
      cv.py
    evaluation/
      survival_metrics.py
      calibration.py
      causal_effects.py
      overlap_balance.py
      bootstrap.py
    inference/
      predict.py
      export.py
  scripts/
    run_train.py
    run_eval.py
    export_model.py
  notebooks/
  mlruns/              # MLflow local
  README.md
```

### 8.2 CLI Entrypoints

The project should expose:

* `python scripts/run_train.py --config configs/dr_learner.yaml`
* `python scripts/run_eval.py --run_id <mlflow_run_id>`
* `python scripts/export_model.py --run_id <id> --output_dir <dir>`

---

## 9. YAML Configuration Specification

### 9.1 Minimal Example

```yaml
project:
  name: prism
  mode: s_learner            # s_learner | t_learner | dr_learner
  model_type: deepsurv       # deepsurv | deephit

data:
  input_format: parquet
  paths:
    demographics: data/raw/demographics.parquet
    labs: data/raw/labs.parquet
    icd10: data/raw/icd10.parquet
    dialysis: data/raw/dialysis.parquet
    mortality: data/raw/mortality.parquet
    followup: data/raw/followup.parquet

cohort:
  egfr_screen_lt15:
    required: true
    outpatient_only: true
    min_days_apart: 90
    max_days_apart: 365
  t0_definition:
    threshold: 10
    outpatient_only: true
    confirmatory:
      enabled: false
      min_days_apart: 14
  exposure:
    early_window_days: 90

features:
  lookback_days: 90
  include:
    - age
    - sex
    - time_since_ckd_days
    - charlson
    - bicarbonate
    - uacr
    - hemoglobin
    - albumin
    - phosphate

preprocessing:
  uacr_from_upcr: true
  imputation:
    method: mice
    max_iter: 10
  scaling:
    log_transform: [creatinine, phosphate, uacr]
    minmax_range: [0.0, 1.0]

training:
  temporal_test_start: "2022-01-01"
  spatial_test_frac: 0.10
  cv:
    folds: 10
    scheme: time_based
  tuning:
    engine: optuna
    n_trials: 50
  hardware:
    gpu: true
    device: cuda:0

dr_learner:
  enabled: false
  propensity_model: gbdt     # logistic | gbdt | xgboost
  trimming:
    clip_min: 0.05
    clip_max: 0.95

evaluation:
  horizons_years: [1, 3, 5]
  bootstrap:
    enabled: true
    n_iterations: 1000
```

### 9.2 Mode-specific Rules

* If `mode: s_learner` → train 1 survival model, include A as an input feature.
* If `mode: t_learner` → train 2 survival models; each model does not require A as input.
* If `mode: dr_learner` → train propensity model, compute weights, train weighted survival model; optionally include A as input (recommended default true).

---

## 10. Deliverables and Acceptance Criteria

### 10.1 Model Deliverables

For each run, produce:

* `model.pt` (PyTorch weights) or pycox model artifact
* `preprocessing.pkl` (sklearn pipeline)
* `config.yaml` (frozen)
* `feature_manifest.json`
* `inference_example.py` showing how to load and predict

In T-learner mode:

* `model_A0.*` and `model_A1.*`

In DR mode:

* `propensity_model.pkl`
* `weights_summary.json` (min/median/max, trimming rate)

### 10.2 Evaluation Deliverables

For each run, produce:

* `metrics.json` with all predictive and causal metrics
* `calibration_1y.png`, `calibration_3y.png`, `calibration_5y.png`
* `overlap_propensity.png` (DR; optional in others)
* `balance_smd.csv` (DR)
* `ate_att_summary.csv`
* `bootstrap_effects.csv`

### 10.3 Quality Gates (Fail/Warning)

Hard fail:

* missing required columns at ingestion
* inability to define t₀ for large fraction of cohort (configurable threshold)
* model artifact not loadable for inference

Warnings (do not fail, but log prominently):

* positivity violations: high proportion with propensity < 0.05 or > 0.95
* extreme weights (e.g., max weight > 50) unless clipped
* post-weighting imbalance: SMD > 0.2 for key covariates

---

## 11. Inference API Requirements

### 11.1 Batch inference interface

`predict_counterfactuals(raw_patient_data, horizons=[1,3,5]) -> DataFrame`

Outputs:

* risk_A0_1y, risk_A0_3y, risk_A0_5y
* risk_A1_1y, risk_A1_3y, risk_A1_5y
* delta_1y, delta_3y, delta_5y

### 11.2 Model portability

The inference package must not require the training dataset; it must carry:

* preprocessing pipeline
* feature manifest
* survival model weights

---

## 12. Implementation Notes (Engineering Decisions)

### 12.1 Why we anchor at t₀ = first eGFR ≤ 10

This is mandated by the protocol to align with common dialysis timing literature and ensure consistent treatment strategy definitions while still allowing recruitment at eGFR < 15. 

### 12.2 Why DR mode is implemented as propensity-weighted survival training

In observational dialysis initiation data, treatment assignment is confounded. Propensity weighting is the most straightforward mechanism to improve comparability of groups on measured covariates and is consistent with causal ML recommendations for robustness. 

---

## 13. Next Steps Checklist (Engineering Execution Plan)

1. Finalize raw data schema mapping (CDARS export → required tables).
2. Implement cohort builder (persistent eGFR < 15 → t₀ eGFR ≤ 10 → A label).
3. Implement feature window extraction and Charlson computation.
4. Implement preprocessing pipeline with saved artifacts.
5. Implement survival model module (DeepSurv baseline; DeepHit optional).
6. Implement learners:

   * S-learner wrapper
   * T-learner wrapper
   * DR-learner wrapper (propensity + weights + weighted training)
7. Implement evaluation suite (predictive + causal + diagnostics).
8. Integrate MLflow logging and ZenML pipeline steps.
9. Create reference configs and smoke tests on a small sample.
10. Run full training and produce reproducible artifact bundles.

---

If you want, I can also generate:

* a **starter repo skeleton** with empty modules and the YAML schema wired up, or
* a **detailed “data dictionary”** template for the CDARS extraction team so the raw tables match the pipeline with minimal manual fixing.
