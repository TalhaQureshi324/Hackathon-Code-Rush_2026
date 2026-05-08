"""
Step 13: Wider Multiplicative Threshold Search
===============================================
WHY THIS FILE:
  - Step 10 found best scales at the EDGE of search range (lower=1.55, upper=0.85).
  - This means the true optimum was OUTSIDE the box.
  - We do a 2-stage search: wide coarse grid -> fine local grid.
  - Multiplicative tuning beat additive tuning for our model (0.7221 vs 0.7189).

WHY RUN IT:
  - This is our last big swing. If we can't beat 0.7221, we submit 0.7221.

COMMAND TO RUN:
  python step13_wider_search.py

OUTPUT:
  submission.csv
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

print("=" * 70)
print("STEP 13: Wider Multiplicative Threshold Search")
print("=" * 70)

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags.csv')
test_bags = pd.read_csv('data/test_bags.csv')
folds = pd.read_csv('data/folds.csv')

train_bags = train_bags.merge(folds, on='bag_id', how='left')
noise_cols = [c for c in train_bags.columns if 'survey_duration_mins' in c]
if noise_cols:
    train_bags = train_bags.drop(columns=noise_cols)
    test_bags = test_bags.drop(columns=noise_cols)

feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label', 'fold']]
X = train_bags[feature_cols]
y = train_bags['label']
X_test = test_bags[feature_cols]

class_weights = {0: 1.1, 1: 1.4, 2: 1.0}
sample_wts = np.array([class_weights[int(v)] for v in y])

# ------------------------------------------------------------------
# 2. QUICK OOF TRAINING (proven models)
# ------------------------------------------------------------------
print("\nRetraining OOF models...")
oof_xgb = np.zeros((len(X), 3))
oof_cbt = np.zeros((len(X), 3))

for fold in range(5):
    print(f"  Fold {fold}...", end=" ")
    tr_mask = train_bags['fold'] != fold
    val_mask = train_bags['fold'] == fold
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    w_tr = sample_wts[tr_mask.values]

    m_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, max_depth=5,
        learning_rate=0.05, n_estimators=2000, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3, reg_alpha=0.5,
        min_child_weight=3, random_state=SEED, n_jobs=4, verbosity=0,
        eval_metric='mlogloss', early_stopping_rounds=100
    )
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_mask.values] = m_xgb.predict_proba(X_val)

    m_cbt = CatBoostClassifier(
        loss_function='MultiClass', eval_metric='TotalF1',
        class_weights=class_weights, depth=6, learning_rate=0.05,
        iterations=2000, early_stopping_rounds=100, l2_leaf_reg=6,
        random_seed=SEED, verbose=False
    )
    m_cbt.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cbt[val_mask.values] = m_cbt.predict_proba(X_val)
    print("Done")

f1_xgb = f1_score(y, oof_xgb.argmax(1), average='macro')
f1_cbt = f1_score(y, oof_cbt.argmax(1), average='macro')
print(f"XGB={f1_xgb:.4f}  CBT={f1_cbt:.4f}")

# Ensemble
best_wx = 0.95
best_wc = 0.05
base_proba = best_wx * oof_xgb + best_wc * oof_cbt
base_f1 = f1_score(y, base_proba.argmax(1), average='macro')
print(f"Base ensemble F1: {base_f1:.4f}")

# ------------------------------------------------------------------
# 3. STAGE 1: COARSE WIDE SEARCH
# ------------------------------------------------------------------
print("\nStage 1: Coarse wide search...")
best_f1 = base_f1
best_s0, best_s1, best_s2 = 1.0, 1.0, 1.0

s0_vals = np.arange(1.0, 3.01, 0.10)   # lower scale
s1_vals = np.arange(0.5, 2.51, 0.10)   # middle scale
s2_vals = np.arange(0.2, 1.51, 0.05)   # upper scale

for s0 in s0_vals:
    for s1 in s1_vals:
        for s2 in s2_vals:
            scaled = base_proba * np.array([s0, s1, s2])
            preds = scaled.argmax(1)
            f1 = f1_score(y, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_s0, best_s1, best_s2 = s0, s1, s2

print(f"Stage 1 best: lower={best_s0:.2f} middle={best_s1:.2f} upper={best_s2:.2f} -> F1={best_f1:.4f}")

# ------------------------------------------------------------------
# 4. STAGE 2: FINE LOCAL SEARCH
# ------------------------------------------------------------------
print("\nStage 2: Fine local search...")
# Search ±0.30 around coarse best with step 0.02
s0_fine = np.arange(max(1.0, best_s0 - 0.30), best_s0 + 0.31, 0.02)
s1_fine = np.arange(max(0.3, best_s1 - 0.30), best_s1 + 0.31, 0.02)
s2_fine = np.arange(max(0.1, best_s2 - 0.30), best_s2 + 0.31, 0.02)

for s0 in s0_fine:
    for s1 in s1_fine:
        for s2 in s2_fine:
            scaled = base_proba * np.array([s0, s1, s2])
            preds = scaled.argmax(1)
            f1 = f1_score(y, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_s0, best_s1, best_s2 = s0, s1, s2

print(f"Stage 2 best: lower={best_s0:.3f} middle={best_s1:.3f} upper={best_s2:.3f} -> F1={best_f1:.4f}")
print(f"Gain over base: +{best_f1 - base_f1:.4f}")

# ------------------------------------------------------------------
# 5. REPORT
# ------------------------------------------------------------------
scaled_best = base_proba * np.array([best_s0, best_s1, best_s2])
best_preds = scaled_best.argmax(1)
print("\n" + classification_report(y, best_preds,
                                   target_names=['lower', 'middle', 'upper'], digits=4))
per_class = f1_score(y, best_preds, average=None)
print(f"lower={per_class[0]:.4f}  middle={per_class[1]:.4f} (tie-break)  upper={per_class[2]:.4f}")

# ------------------------------------------------------------------
# 6. FINAL TRAIN + SUBMIT
# ------------------------------------------------------------------
print("\nTraining final models...")
final_xgb = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3, max_depth=5,
    learning_rate=0.05, n_estimators=500, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=3, reg_alpha=0.5,
    min_child_weight=3, random_state=SEED, n_jobs=4, verbosity=0
)
final_xgb.fit(X, y, sample_weight=sample_wts, verbose=False)

final_cbt = CatBoostClassifier(
    loss_function='MultiClass', class_weights=class_weights,
    depth=6, learning_rate=0.05, iterations=800,
    l2_leaf_reg=6, random_seed=SEED, verbose=False
)
final_cbt.fit(X, y, verbose=False)

test_proba = 0.95 * final_xgb.predict_proba(X_test) + 0.05 * final_cbt.predict_proba(X_test)
test_proba[:, 0] *= best_s0
test_proba[:, 1] *= best_s1
test_proba[:, 2] *= best_s2
test_preds = test_proba.argmax(1)

submission = pd.DataFrame({'bag_id': test_bags['bag_id'].values, 'label': test_preds})
submission.to_csv('submission.csv', index=False)

print("\n" + "=" * 70)
print("FINAL SUBMISSION GENERATED")
print("=" * 70)
print(f"Expected Macro F1: {best_f1:.4f}")
print(f"Scales: lower={best_s0:.3f} middle={best_s1:.3f} upper={best_s2:.3f}")
print("\n>>> UPLOAD TO KAGGLE <<<")
