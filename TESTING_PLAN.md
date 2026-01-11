# PRISM Testing and Debugging Plan

**Date**: 2026-01-11
**Status**: Ready to begin testing

---

## Testing Strategy

We will test the pipeline step-by-step, starting from data ingestion through to final evaluation. Each step will be tested independently before integration testing.

---

## Phase 1: Module-Level Testing

### ✅ Test 1: Data Ingestion (Existing)
**Module**: `steps/ingest_data.py`
**Status**: Should work (existing from previous project)
**Test**:
```python
from steps.ingest_data import ingest_data
lab_dfs = ingest_data()
print(f"Keys: {lab_dfs.keys()}")
print(f"Creatinine shape: {lab_dfs['cr_df'].shape}")
```

### 🔄 Test 2: Cohort Formation
**Module**: `src/cohort_builder.py`, `steps/form_cohort.py`
**Dependencies**: creatinine (for eGFR), operation_df, death_df
**Test**:
```python
from src.cohort_builder import CohortBuilder
builder = CohortBuilder()
cohort_df = builder.build_cohort(cr_df, operation_df, death_df)
print(cohort_df.head())
print(f"Cohort size: {len(cohort_df)}")
print(f"Treatment balance: {cohort_df['A'].value_counts()}")
```

**Expected Issues**:
- eGFR calculation may be missing or in wrong place
- Column names may not match (need to check cr_df structure)
- Date formats may need conversion

### 🔄 Test 3: Feature Extraction
**Module**: `src/feature_extractor.py`, `steps/extract_features.py`
**Dependencies**: cohort_df, lab_dfs, icd10_df
**Test**:
```python
from src.feature_extractor import FeatureExtractor
extractor = FeatureExtractor()
features_df = extractor.extract(cohort_df, lab_dfs, icd10_df)
print(features_df.head())
print(f"Features: {len(features_df.columns)}")
```

**Expected Issues**:
- Lab DataFrame column names may not match
- ICD-10 processing may need adjustment
- Missing value handling

### 🔄 Test 4: Preprocessing
**Module**: `src/ckd_preprocessor.py`, `steps/preprocess_data.py`
**Dependencies**: master_df (merged cohort + features)
**Test**:
```python
from src.ckd_preprocessor import CKDPreprocessor
preprocessor = CKDPreprocessor()
train_processed = preprocessor.fit_transform(train_df)
```

**Expected Issues**:
- Column type mismatches
- MICE imputation may need tuning

### 🔄 Test 5: S-Learner
**Module**: `src/s_learner.py`, `steps/train_s_learner.py`
**Dependencies**: preprocessed data
**Test**:
```python
from src.s_learner import SLearner
learner = SLearner(input_dim=X.shape[1]+1)
learner.fit(X, A, durations, events)
ate = learner.compute_ate(X)
print(f"ATE: {ate}")
```

**Expected Issues**:
- PyTorch/CUDA compatibility
- Input dimension mismatches
- Missing dependencies (pycox, torchtuples)

### 🔄 Test 6: T-Learner
**Module**: `src/t_learner.py`
**Expected Issues**:
- Sample size in treatment groups
- Model convergence

### 🔄 Test 7: DR-Learner
**Module**: `src/dr_learner.py`, `src/propensity_model.py`
**Expected Issues**:
- Propensity score extremes
- Weight computation
- Weighted training implementation

### 🔄 Test 8: Evaluation
**Module**: `src/causal_evaluator.py`, `steps/evaluate_learner.py`
**Expected Issues**:
- Metric computation
- Plot generation
- MLflow logging

---

## Phase 2: Integration Testing

### 🔄 Test 9: End-to-End Pipeline (Smoke Test)
```bash
python scripts/run_prism.py \
    --config configs/s_learner.yaml \
    --subset 100 \
    --epochs 2 \
    --experiment smoke_test \
    --dry-run
```

### 🔄 Test 10: Full S-Learner Run
```bash
python scripts/run_prism.py \
    --config configs/s_learner.yaml \
    --subset 1000 \
    --epochs 10
```

---

## Phase 3: Full Training

### 🔄 Test 11: Complete Dataset - S-Learner
```bash
python scripts/run_prism.py --config configs/s_learner.yaml
```

### 🔄 Test 12: Complete Dataset - T-Learner
```bash
python scripts/run_prism.py --config configs/t_learner.yaml
```

### 🔄 Test 13: Complete Dataset - DR-Learner
```bash
python scripts/run_prism.py --config configs/dr_learner.yaml
```

---

## Common Issues to Watch For

### Data Issues
- [ ] Missing eGFR calculation
- [ ] Date format inconsistencies
- [ ] Column name mismatches
- [ ] Missing value patterns

### Module Issues
- [ ] Import errors
- [ ] Missing dependencies
- [ ] Function signature mismatches
- [ ] Type errors

### Pipeline Issues
- [ ] Step ordering
- [ ] Data passing between steps
- [ ] Configuration parameter parsing

### Model Issues
- [ ] CUDA/CPU compatibility
- [ ] Memory errors
- [ ] Convergence problems
- [ ] NaN losses

---

## Testing Checklist

### Before Testing
- [x] Code pushed to GitHub
- [ ] Python environment set up
- [ ] Dependencies installed
- [ ] Data files available

### During Testing
- [ ] Test each module independently
- [ ] Document all errors
- [ ] Fix errors incrementally
- [ ] Re-test after fixes

### After Testing
- [ ] All modules pass unit tests
- [ ] Integration test passes
- [ ] Full pipeline runs successfully
- [ ] Results are sensible

---

## Next Steps

1. **Start with Test 1**: Verify data ingestion works
2. **Progress sequentially**: Test each module in order
3. **Fix issues immediately**: Don't accumulate technical debt
4. **Document solutions**: Keep track of what was fixed

Ready to begin? Let's start with Test 1!
