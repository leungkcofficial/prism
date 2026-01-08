# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PRISM (Predictive Renal Intelligence Survival Modeling)** is a causal survival analysis project aimed at building deep-learning models to estimate individualized all-cause mortality risk for patients with advanced CKD at the dialysis decision point.

### Key Objectives
- Predict mortality risk under two treatment strategies: early dialysis (A=1) vs non-early (A=0)
- Support three causal learning modes: **S-learner**, **T-learner**, and **DR-learner** (propensity-weighted)
- Index date (t₀) defined as first outpatient eGFR ≤ 10 mL/min/1.73m² after screening eligibility (persistent eGFR < 15)
- Early dialysis = initiation within 90 days after t₀

**Critical Note**: The existing codebase contains a **data pipeline from a different CKD project**. The data ingestion components are reusable, but the cohort formation, feature extraction, and modeling logic must be redesigned to match the PRISM specifications in `doc/PRD_main.md`.

## Repository Structure

```
prism/
├── doc/PRD_main.md              # Full project specification (READ THIS FIRST)
├── data/                        # Raw EHR data (gitignored)
│   ├── Cr/, Hb/, alb/, po4/    # Lab results (quarterly CSV files)
│   ├── icd10/, death/, ot/     # Clinical data
│   └── a1c/, ca/, hco3/, upacr/ # Additional labs
├── src/                         # Core data processing modules
│   ├── data_ingester.py        # CSV file loading
│   ├── data_mapper.py          # Factory for creating type-specific mappers
│   ├── lab_result_mapper.py    # Lab result processing & validation
│   ├── dx_ingester.py          # ICD-10 diagnosis processing (CCI calculation)
│   ├── death_ingester.py       # Mortality data processing
│   ├── ot_ingester.py          # Operation/dialysis data processing
│   ├── data_cleaning.py        # Data cleaning utilities
│   ├── duration_mapper.py      # Survival time discretization for DeepHit
│   ├── temporal_merge.py       # Temporal data merging utilities
│   ├── ckd_preprocessor.py     # Production preprocessing class
│   └── *.yml                   # Configuration files for data types, validation, mappings
├── steps/                       # ZenML pipeline steps (from previous project)
│   ├── ingest_data.py          # Data ingestion step (REUSABLE)
│   ├── merge_data.py           # Data merging step (NEEDS REDESIGN)
│   ├── preprocess_data.py      # Preprocessing step (NEEDS REDESIGN)
│   ├── split_data.py           # Train/test splitting (NEEDS REDESIGN)
│   ├── impute_data.py          # MICE imputation step
│   └── EDA.py                  # Exploratory data analysis
└── run_pipeline.py             # Pipeline orchestration (from previous project)
```

## Data Architecture

### Raw Data Format
- **Lab data**: Quarterly CSV files with columns: `Reference Key, Date of Birth, Sex, LIS Reference Datetime, LIS Case No., LIS Result`
- **Clinical data**: CSV files with patient key, dates, and event-specific columns
- Data spans 2009-2023, organized by quarters (e.g., `RRT2009q12.csv`)

### Data Types Configuration
Data types are defined in `src/data_types.yml`:
- **Lab data**: creatinine, hemoglobin, a1c, albumin, phosphate, calcium, bicarbonate, UPCR, UACR
- **Clinical data**: ICD-10 codes, death records, operations (including dialysis)

### Data Processing Pipeline (Current State)

The existing pipeline follows this flow:
1. **Ingestion** (`ingest_data.py`): Loads 14 DataFrames (10 lab types, demographics, ICD-10, death, operations)
2. **Merging** (`merge_data.py`): Creates temporal key-date pairs, forward-fills lab values
3. **Cleaning** (`data_cleaning.py`): Removes inpatient records, handles RRT clinic codes
4. **Imputation** (`impute_data.py`): MICE imputation for missing values
5. **Preprocessing** (`preprocess_data.py`): Log transformation, min-max scaling, feature engineering
6. **Splitting** (`split_data.py`): Temporal (post-2022) + spatial (random 10%) test sets

### Key Design Patterns

**Factory Pattern**: `DataMapperFactory` creates appropriate data mappers based on data type
```python
mapper = DataMapperFactory.create_mapper('creatinine', validation_rules)
lab_df, demo_df = mapper.process(raw_data)
```

**Strategy Pattern**: Different mappers (`StandardLabResultMapper`, `UrineLabResultMapper`, `CalciumLabResultMapper`) implement the `DataMapper` interface

**ZenML Integration**: All pipeline steps are decorated with `@step` for orchestration and artifact tracking

## Critical Gaps vs PRD Requirements

| **Component** | **Status** | **Action Needed** |
|--------------|-----------|-------------------|
| Cohort formation (eGFR screening, t₀ definition) | ❌ Missing | Build new module |
| Treatment labeling (early vs non-early dialysis) | ⚠️ Partial | Extract from `ot_ingester.py`, add 90-day logic |
| Index-date feature extraction (lookback windows) | ❌ Wrong approach | Replace forward-fill with t₀-centric windowing |
| Survival outcome (time-to-death from t₀) | ⚠️ Partial | Align with t₀, 5-year censoring |
| eGFR calculation (CKD-EPI) | ⚠️ Unclear | Verify formula in `data_cleaning.py` |
| Survival models (DeepSurv/DeepHit) | ❌ Missing | Implement with pycox |
| Causal learners (S/T/DR) | ❌ Missing | Implement all three modes |
| Propensity weighting | ❌ Missing | Build propensity model for DR-learner |

## Configuration Files

### Environment Variables
Expected variables (typically in `.env` file, not committed):
- `STUDY_END_DATE`: Study end date for censoring (default: "2023-12-31")
- `TEMPORAL_CUTOFF_DATE`: Temporal test split date (default: "2022-01-01")
- `SPATIAL_TEST_RATIO`: Spatial test split ratio (default: 0.10)
- `RANDOM_SEED`: Random seed for reproducibility (default: 42)
- `EDA_OUTPUT_PATH`: Output path for EDA results (default: "results")
- `HARD_TRUTH_COLUMNS`, `MED_HISTORY_PATTERN`, `LAB_COLUMNS`, etc.: Column categorizations

### YAML Configurations
- `src/data_types.yml`: Maps data types to directory names
- `src/default_data_validation_rules.yml`: Min/max validation rules for lab values
- `src/default_master_df_mapping.yml`: Feature mappings for survival models (cluster, duration, event, features)
- `src/default_ingest_data_output_dataframe structure.yml`: Expected output schema from ingestion
- `src/default_clean_data_output_dataframe_structure.yml`: Expected schema after cleaning

## Important Implementation Notes

### Data Validation Rules
Lab results are validated against ranges defined in `default_data_validation_rules.yml`. Invalid values are set to NaN.

### Charlson Comorbidity Index (CCI)
- ICD-10 codes are mapped to CCI categories in `dx_ingester.py`
- Binary flags for each comorbidity (e.g., `myocardial_infarction`, `diabetes_wo_complication`)
- Total CCI score computed as `cci_score_total`

### UACR Derivation
If UACR is missing, it can be derived from UPCR using published conversion formulas (implemented in `lab_result_mapper.py`)

### Dialysis Detection
- RRT (Renal Replacement Therapy) codes in creatinine data indicate clinic visits
- Operation data (`ot_ingester.py`) contains `is_dialysis` flag
- **Critical**: Current code doesn't implement the 90-day window for early vs non-early classification

### eGFR Calculation
- Should use CKD-EPI formula: requires creatinine, age, sex
- Current implementation in `data_cleaning.py` needs verification
- Must filter to outpatient records only to avoid AKI contamination

### Temporal Merging Strategy
The current approach uses forward-fill within patient groups, which is **NOT suitable for the PRISM project**. The PRD requires:
- Features extracted using lookback windows from t₀ (e.g., closest lab value within 90 days before t₀)
- One row per patient at t₀ with all features, treatment A, and survival outcome

## Modeling Architecture (To Be Implemented)

### Survival Backend
- **Primary**: DeepSurv (single-event mortality) via `pycox`
- **Optional**: DeepHit (competing risks) via `pycox`
- Requires discrete time intervals (use `duration_mapper.py` for DeepHit)

### Causal Learner Modes
1. **S-learner**: Single model f(X, A), counterfactuals by setting A=0 vs A=1
2. **T-learner**: Two models f₀(X) and f₁(X), trained on A=0 and A=1 subsets
3. **DR-learner**: Propensity model e(X) + weighted survival model

### Evaluation Framework (To Be Implemented)
- **Predictive metrics**: C-index, Brier score, calibration curves at 1/3/5 years
- **Causal metrics**: ATE/ATT at 1/3/5 years with bootstrap CI
- **Overlap diagnostics**: Propensity distributions, trimming stats (for DR)
- **Balance diagnostics**: SMD pre/post weighting (for DR)

## Development Workflow

### Working with the Current Codebase
1. **Reusable components**: Data ingestion, lab mappers, CCI calculation, UACR conversion
2. **Needs redesign**: Cohort formation, feature extraction, preprocessing, splitting
3. **Missing entirely**: Survival modeling, causal learners, evaluation

### Next Steps (As Per PRD)
1. Build cohort formation module (eGFR screening → t₀ definition → treatment labeling)
2. Implement index-date feature extraction with lookback windows
3. Create survival outcome variables (duration from t₀, event indicator)
4. Implement DeepSurv wrapper and causal learner modes
5. Build evaluation suite (predictive + causal metrics)

### Data Pipeline Best Practices
- Always use `key` as patient identifier (integer)
- Always use `date` as datetime column for temporal operations
- Forward-fill missing values **only within patient groups**, sorted by date
- Preserve original columns when creating derived features
- Use `.copy()` to avoid modifying original DataFrames

### ZenML Pipeline Execution
The current `run_pipeline.py` orchestrates steps using ZenML. When building new steps:
- Decorate functions with `@step`
- Clearly specify input/output types
- Return DataFrames (not None) even on errors
- Log progress and intermediate statistics

## Known Issues and Limitations

1. **No cohort formation logic**: The existing code assumes patients are already selected and t₀ is defined
2. **Forward-fill approach**: Current merging uses forward-fill, not lookback windows from an index date
3. **Endpoint definition mismatch**: Current `endpoint` variable doesn't align with t₀-based survival analysis
4. **No treatment variable**: No logic to define early vs non-early dialysis (A=1 vs A=0)
5. **Incomplete eGFR pipeline**: eGFR calculation exists but needs verification against CKD-EPI formula
6. **Missing survival models**: No DeepSurv/DeepHit implementation exists
7. **No causal inference**: S/T/DR learners not implemented
8. **Test split issues**: Current temporal/spatial splits don't account for t₀-based cohort structure

## Reading the PRD

The `doc/PRD_main.md` file is **mandatory reading** for understanding:
- Cohort formation logic (persistent eGFR <15, t₀ at eGFR ≤10)
- Treatment definition (90-day window for early dialysis)
- Feature extraction windows (90 days for labs, 5 years for CCI)
- Survival outcome definition (time-to-death from t₀, 5-year censoring)
- Causal learner specifications (S/T/DR modes)
- Evaluation requirements (predictive + causal metrics)
- Sensitivity analyses (varying early window, t₀ threshold, trimming)

## Key References in Code

- **Data ingestion flow**: `steps/ingest_data.py` → `src/data_ingester.py` → `src/lab_result_mapper.py`
- **CCI calculation**: `src/dx_ingester.py` (ICD-10 to CCI mapping)
- **Dialysis detection**: `src/ot_ingester.py` (`is_dialysis` flag)
- **UACR conversion**: `src/lab_result_mapper.py` (`UrineLabResultMapper`)
- **Survival discretization**: `src/duration_mapper.py` (for DeepHit)
- **Preprocessing pipeline**: `src/ckd_preprocessor.py` (production-ready class)
