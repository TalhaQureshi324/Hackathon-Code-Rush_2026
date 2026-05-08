"""
Step 10: Threshold Tuning for Macro F1
========================================
WHY THIS FILE:
  - Your partner jumped from 0.7107 -> 0.7261 using threshold tuning.
  - Default argmax assumes all classes are equal. But lower class is under-predicted.
  - By scaling class probabilities before argmax, we can boost lower/middle recall.
  - This is the single biggest remaining upgrade.

WHY RUN IT:
  - Should jump your score from ~0.71 to ~0.72-0.73.
  - Generates the FINAL submission.csv.

COMMAND TO RUN:
  python step10_threshold_tune.py

OUTPUT:
  submission.csv  (overwritten with threshold-tuned predictions)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb
from catboost import CatBoostClassifier

print("=" * 70)
print("STEP 10: Threshold Tuning for Maximum Macro F1")
print("=" * 70)

# ------------------------------------------------------------------
# 1. LOAD DATA AND BEST OOF PROBABILITIES
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags.csv')
folds = pd.read_csv('data/folds.csv')
train_bags = train_bags.merge(folds, on='bag_id', how='left')

noise_cols = [c for c in train_bags.columns if 'survey_duration_mins' in c]
if noise_cols:
    train_bags = train_bags.drop(columns=noise_cols)

feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label', 'fold']]
X = train_bags[feature_cols]
y = train_bags['label']

class_weights = {0: 1.1, 1: 1.4, 2: 1.0}

# ------------------------------------------------------------------
# 2. REGENERATE OOF PROBABILITIES (XGBoost + CatBoost ensemble)
# ------------------------------------------------------------------
print("Regenerating OOF probabilities...")

xgb_oof_proba = np.zeros((len(train_bags), 3))
cat_oof_proba = np.zeros((len(train_bags), 3))

for fold in range(5):
    print(f"  Fold {fold}...", end=" ")
    tr_mask = train_bags['fold'] != fold
    val_mask = train_bags['fold'] == fold

    # XGBoost
    sample_weight = np.array([class_weights[int(v)] for v in y[tr_mask]])
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, max_depth=5,
        learning_rate=0.05, n_estimators=500, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3, reg_alpha=0.5,
        min_child_weight=3, random_state=42, n_jobs=4, verbosity=0
    )
    xgb_model.fit(X[tr_mask], y[tr_mask], sample_weight=sample_weight, verbose=False)
    xgb_oof_proba[val_mask] = xgb_model.predict_proba(X[val_mask])

    # CatBoost
    cat_model = CatBoostClassifier(
        loss_function='MultiClass', eval_metric='TotalF1',
        class_weights=class_weights, depth=6, learning_rate=0.05,
        iterations=800, l2_leaf_reg=6, random_seed=42, verbose=False
    )
    cat_model.fit(X[tr_mask], y[tr_mask], eval_set=(X[val_mask], y[val_mask]), verbose=False)
    cat_oof_proba[val_mask] = cat_model.predict_proba(X[val_mask])

    print("Done")

# Best ensemble weight from Step 9
ensemble_oof_proba = 0.95 * xgb_oof_proba + 0.05 * cat_oof_proba
ensemble_preds = ensemble_oof_proba.argmax(axis=1)
base_f1 = f1_score(y, ensemble_preds, average='macro')
print(f"\nBase ensemble (no tuning) Macro F1: {base_f1:.4f}")

# ------------------------------------------------------------------
# 3. THRESHOLD TUNING: Scale probabilities per class
# ------------------------------------------------------------------
print("\nSearching best probability scales...")

best_f1 = base_f1
best_scales = [1.0, 1.0, 1.0]

# Coarse search: boost lower and middle, keep upper near 1.0
# We know lower is under-predicted, so s0 > 1.0 helps
# We know middle is the tie-breaker, so s1 around 1.0-1.3 helps
# Upper is already strong, so s2 near 1.0 is fine

count = 0
for s0 in np.arange(0.9, 1.6, 0.05):      # lower scale
    for s1 in np.arange(0.9, 1.4, 0.05):  # middle scale
        for s2 in np.arange(0.8, 1.2, 0.05):  # upper scale
            scaled = ensemble_oof_proba * np.array([s0, s1, s2])
            preds = scaled.argmax(axis=1)
            f1 = f1_score(y, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_scales = [s0, s1, s2]
            count += 1

print(f"Searched {count} combinations.")
print(f"\n>>> BEST SCALES: lower={best_scales[0]:.2f}, middle={best_scales[1]:.2f}, upper={best_scales[2]:.2f} <<<")
print(f">>> TUNED Macro F1: {best_f1:.4f} <<<")
print(f">>> GAIN: +{best_f1 - base_f1:.4f} <<<")

# Per-class breakdown with best scales
scaled_best = ensemble_oof_proba * np.array(best_scales)
best_preds = scaled_best.argmax(axis=1)
from sklearn.metrics import classification_report
print("\nPer-class F1 (tuned):")
print(classification_report(y, best_preds,
                            target_names=['lower (0)', 'middle (1)', 'upper (2)'],
                            digits=4))

# ------------------------------------------------------------------
# 4. TRAIN FINAL MODELS + PREDICT TEST WITH TUNED THRESHOLDS
# ------------------------------------------------------------------
print("\nTraining final models on 100% data...")

# Final XGBoost
sample_weight = np.array([class_weights[int(v)] for v in y])
final_xgb = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3, max_depth=5,
    learning_rate=0.05, n_estimators=500, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=3, reg_alpha=0.5,
    min_child_weight=3, random_state=42, n_jobs=4, verbosity=0
)
final_xgb.fit(X, y, sample_weight=sample_weight)

# Final CatBoost
final_cat = CatBoostClassifier(
    loss_function='MultiClass', class_weights=class_weights,
    depth=6, learning_rate=0.05, iterations=800,
    l2_leaf_reg=6, random_seed=42, verbose=False
)
final_cat.fit(X, y, verbose=False)

# Predict test
test_bags = pd.read_csv('data/test_bags.csv')
test_bags = test_bags.drop(columns=noise_cols, errors='ignore')
X_test = test_bags[feature_cols]

xgb_test_proba = final_xgb.predict_proba(X_test)
cat_test_proba = final_cat.predict_proba(X_test)
ensemble_test_proba = 0.95 * xgb_test_proba + 0.05 * cat_test_proba

# Apply tuned scales
tuned_test_proba = ensemble_test_proba * np.array(best_scales)
test_preds = tuned_test_proba.argmax(axis=1)

# ------------------------------------------------------------------
# 5. SAVE FINAL SUBMISSION
# ------------------------------------------------------------------
submission = pd.DataFrame({
    'bag_id': test_bags['bag_id'].values,
    'label': test_preds
})
submission.to_csv('submission.csv', index=False)

print("\n" + "=" * 70)
print("FINAL THRESHOLD-TUNED SUBMISSION GENERATED")
print("=" * 70)
print(f"File: submission.csv")
print(f"Shape: {submission.shape}")
print(f"Label distribution:")
print(submission['label'].value_counts().sort_index())
print(f"\nTuned scales: lower={best_scales[0]:.3f}, middle={best_scales[1]:.3f}, upper={best_scales[2]:.3f}")
print(f"Expected Macro F1: {best_f1:.4f}")
print("\n>>> UPLOAD TO KAGGLE NOW <<<")
