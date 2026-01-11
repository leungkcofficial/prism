# PRISM Development Status

**Last Updated:** 2026-01-11
**Current Phase:** Initial Implementation (Week 1-3 of development plan)

## ✅ Completed Components

### Phase 1: Setup & TAROT2 Integration (Week 1)
- [x] **Copied 7 core TAROT2 modules to `src/`:**
  - `nn_architectures.py` (23K) - Neural network architectures (MLP, CNN-MLP, LSTM)
  - `survival_utils.py` (26K) - PyCox utilities for survival analysis
  - `metric_calculator.py` (86K) - Comprehensive evaluation metrics
  - `eval_model.py` (57K) - Full evaluation pipeline
  - `ckd_preprocessor.py` (23K) - Production preprocessing class
  - `util.py` (59K) - General utilities (YAML, HDF5, CUDA)
  - `hyperparameter_config.yml` (2.1K) - Hyperparameter search spaces

- [x] **Copied 2 training steps from TAROT2 to `steps/`:**
  - `model_train_surv.py` (14K) - DeepSurv training step
  - `hyperparameter_optimization.py` (65K) - Optuna HPO step

- [x] **Created directory structure:**
  - `configs/` - Configuration files
  - `configs/sensitivity/` - Sensitivity analysis configs
  - `scripts/` - CLI entry points

### Phase 2: Cohort Formation (Week 2)
- [x] **Created `src/cohort_builder.py`** (14K)
  - Persistent eGFR <15 screening logic
  - Index date (t₀) definition (first eGFR ≤10)
  - 90-day early dialysis labeling
  - Survival outcome calculation
  - Comprehensive validation and logging

- [x] **Created `steps/form_cohort.py`** (5K)
  - ZenML step wrapper for CohortBuilder
  - Configuration management
  - Summary statistics logging

### Phase 3: Feature Extraction (Week 3)
- [x] **Created `src/feature_extractor.py`** (11K)
  - t₀-centric lab feature extraction (90-day lookback)
  - CCI feature extraction (5-year lookback)
  - UACR derivation from UPCR
  - Time since CKD onset calculation
  - Efficient temporal joins using pandas merge_asof

- [x] **Created `steps/extract_features.py`** (4K)
  - ZenML step wrapper for FeatureExtractor
  - Configuration management
  - Summary statistics logging

### Phase 4: Data Pipeline Integration
- [x] **Created `steps/merge_cohort_features.py`** (3K)
  - Merges cohort outcomes with extracted features
  - Validation and quality checks

### Phase 5: Configuration Files
- [x] **Created `configs/s_learner.yaml`** (5K)
  - Complete configuration for S-learner mode
  - Cohort, features, model, evaluation parameters

- [x] **Created `configs/t_learner.yaml`** (5K)
  - Complete configuration for T-learner mode
  - Two separate model configurations

- [x] **Created `configs/dr_learner.yaml`** (6K)
  - Complete configuration for DR-learner mode
  - Propensity model and IPTW parameters
  - Balance diagnostics configuration

- [x] **Created sensitivity analysis configs:**
  - `configs/sensitivity/early_60d.yaml` - 60-day early window
  - `configs/sensitivity/early_120d.yaml` - 120-day early window
  - `configs/sensitivity/t0_egfr12.yaml` - eGFR ≤12 threshold

## 🚧 In Progress / Next Steps

### Week 4-5: Causal Learners & Survival Models
- [ ] **Create `src/deepsurv_wrapper.py`** (Priority 1)
  - Adapt TAROT2's DeepSurv training loop
  - Add weighted training for DR-learner
  - Interface with copied `nn_architectures.py`

- [ ] **Create `src/s_learner.py`** (Priority 2)
  - S-learner implementation
  - Counterfactual prediction
  - ATE/ATT calculation

- [ ] **Create `src/t_learner.py`** (Priority 3)
  - T-learner implementation
  - Two separate model training
  - Sample size validation

- [ ] **Create `src/dr_learner.py`** (Priority 4)
  - DR-learner implementation
  - Integration with propensity model

- [ ] **Create `src/propensity_model.py`** (Priority 4)
  - Propensity score estimation
  - IPTW weight computation
  - Overlap diagnostics

### Week 5-6: Evaluation Framework
- [ ] **Create `src/causal_evaluator.py`**
  - ATE/ATT with bootstrap CI
  - Overlap diagnostics
  - Balance metrics (SMD)
  - Integration with TAROT2's metric_calculator.py

- [ ] **Create `steps/evaluate_learner.py`**
  - ZenML step wrapper for evaluation
  - MLflow logging integration

### Week 6-7: Pipeline Integration
- [ ] **Create training steps:**
  - `steps/train_s_learner.py`
  - `steps/train_t_learner.py`
  - `steps/train_dr_learner.py`

- [ ] **Create `pipelines/prism_training_pipeline.py`**
  - Full pipeline orchestration
  - ZenML integration
  - MLflow experiment tracking

- [ ] **Create `scripts/run_prism.py`**
  - CLI entry point
  - Configuration loading
  - Pipeline execution

### Week 7-8: Testing & Validation
- [ ] **Create unit tests:**
  - `tests/test_cohort_builder.py`
  - `tests/test_feature_extractor.py`
  - `tests/test_learners.py`
  - `tests/test_evaluation.py`

- [ ] **Create integration tests:**
  - `tests/test_full_pipeline.py`

- [ ] **Run smoke test** with 1000-patient subset

- [ ] **Run full training** for all three modes

- [ ] **Run sensitivity analyses**

## 📊 Architecture Overview

### Current Folder Structure
```
prism/
├── src/                          # Core logic (functions & classes)
│   ├── cohort_builder.py        ✅ NEW
│   ├── feature_extractor.py     ✅ NEW
│   ├── deepsurv_wrapper.py      ⏳ TODO
│   ├── s_learner.py             ⏳ TODO
│   ├── t_learner.py             ⏳ TODO
│   ├── dr_learner.py            ⏳ TODO
│   ├── propensity_model.py      ⏳ TODO
│   ├── causal_evaluator.py      ⏳ TODO
│   ├── nn_architectures.py      ✅ FROM TAROT2
│   ├── survival_utils.py        ✅ FROM TAROT2
│   ├── metric_calculator.py     ✅ FROM TAROT2
│   ├── eval_model.py            ✅ FROM TAROT2
│   ├── ckd_preprocessor.py      ✅ FROM TAROT2
│   ├── util.py                  ✅ FROM TAROT2
│   └── hyperparameter_config.yml ✅ FROM TAROT2
│
├── steps/                        # ZenML pipeline steps
│   ├── form_cohort.py           ✅ NEW
│   ├── extract_features.py      ✅ NEW
│   ├── merge_cohort_features.py ✅ NEW
│   ├── train_s_learner.py       ⏳ TODO
│   ├── train_t_learner.py       ⏳ TODO
│   ├── train_dr_learner.py      ⏳ TODO
│   ├── evaluate_learner.py      ⏳ TODO
│   ├── model_train_surv.py      ✅ FROM TAROT2
│   └── hyperparameter_optimization.py ✅ FROM TAROT2
│
├── pipelines/                    # Pipeline orchestration
│   └── prism_training_pipeline.py ⏳ TODO
│
├── configs/                      # Configuration files
│   ├── s_learner.yaml           ✅ NEW
│   ├── t_learner.yaml           ✅ NEW
│   ├── dr_learner.yaml          ✅ NEW
│   └── sensitivity/
│       ├── early_60d.yaml       ✅ NEW
│       ├── early_120d.yaml      ✅ NEW
│       └── t0_egfr12.yaml       ✅ NEW
│
├── scripts/                      # CLI entry points
│   └── run_prism.py             ⏳ TODO
│
├── doc/                          # Documentation
│   └── PRD_main.md              ✅ EXISTING
│
└── data/                         # Raw EHR data (gitignored)
```

## 🎯 Key Design Decisions

1. **Folder Structure:**
   - `src/` contains all core logic (functions and classes)
   - `steps/` contains ZenML wrappers that call `src/` functions
   - `pipelines/` contains workflow orchestration

2. **DeepSurv Only:**
   - Single endpoint (all-cause mortality) means DeepHit is NOT needed
   - Focus on Cox Proportional Hazards model

3. **TAROT2 Reuse:**
   - ~70% of survival modeling infrastructure reused from TAROT2
   - ~30% new development (cohort formation, t₀-centric features, causal learners)

4. **Configuration Management:**
   - YAML-based configuration for all three learner modes
   - Separate sensitivity analysis configs
   - Easy to modify parameters without changing code

## 📝 Notes

### Dependencies from TAROT2
The copied TAROT2 modules have dependencies that need to be verified:
- `pycox` - Survival analysis with PyTorch
- `torch`, `torchtuples` - PyTorch ecosystem
- `scikit-survival` - Additional survival utilities
- `optuna` - Hyperparameter optimization
- `mlflow` - Experiment tracking
- `zenml` - Pipeline orchestration

### Integration Points
1. **Cohort Formation → Feature Extraction:**
   - `cohort_df` contains `t0_date` used as reference for lookback windows
   - One row per patient at index date

2. **Feature Extraction → Preprocessing:**
   - Features extracted at t₀ are then imputed and scaled
   - Reuse TAROT2's `ckd_preprocessor.py`

3. **Preprocessing → Training:**
   - Preprocessed features + treatment A + survival outcomes
   - Feed into causal learners (S/T/DR)

4. **Training → Evaluation:**
   - Trained models generate counterfactual predictions
   - Evaluation computes predictive + causal metrics

## 🚀 Quick Start (Once Complete)

```bash
# S-learner
python scripts/run_prism.py --config configs/s_learner.yaml

# T-learner
python scripts/run_prism.py --config configs/t_learner.yaml

# DR-learner
python scripts/run_prism.py --config configs/dr_learner.yaml

# Sensitivity analysis
python scripts/run_prism.py --config configs/sensitivity/early_60d.yaml
```

## 📊 Expected Timeline

- **Week 1-3:** ✅ Setup, cohort formation, feature extraction (COMPLETE)
- **Week 4-5:** ⏳ Causal learners & survival models (IN PROGRESS)
- **Week 5-6:** ⏳ Evaluation framework
- **Week 6-7:** ⏳ Pipeline integration
- **Week 7-8:** ⏳ Testing & production

**Total Duration:** 8 weeks → **5-6 weeks with TAROT2 reuse**

## 📧 Contact

For questions or issues, refer to:
- `doc/PRD_main.md` - Full project specification
- `CLAUDE.md` - Project instructions for Claude Code
- `/home/goma/.claude/plans/abundant-soaring-trinket.md` - Detailed development plan
