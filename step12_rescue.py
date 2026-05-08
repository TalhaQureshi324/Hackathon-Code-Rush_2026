"""
Step 12: The Rescue — Proven Models + Additive Threshold Tuning
================================================================
WHY THIS FILE:
  - Step 11 proved partner's hyperparameters DON'T work on our rich features.
  - We go BACK to our proven models:
      XGBoost: depth=5, lr=0.05, reg_lambda=3  -> 0.7086 OOF
      CatBoost: depth=6, lr=0.05, l2_leaf_reg=6 -> 0.6827 OOF
  - But we use ADDITIVE threshold tuning (stolen from partner) instead of
    our weaker multiplicative scaling.
  - Wider search range (±0.15) to find the true optimum.

WHY RUN IT:
  - This should beat our previous best of 0.7221.
  - Target: 0.725–0.730.

COMMAND TO RUN:
  python step12_rescue.py

OUTPUT:
  submission.csv
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

print("=" * 70)
print("STEP 12: Rescue — Proven Models + Additive Thresholds")
print("=" * 70)

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. LOAD CLEAN DATA
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

print(f"Features: {len(feature_cols)}")

# OUR proven class weights (NOT partner's)
class_weights = {0: 1.1, 1: 1.4, 2: 1.0}
sample_wts = np.array([class_weights[int(v)] for v in y])

# ------------------------------------------------------------------
# 2. OOF TRAINING — PROVEN PARAMS
# ------------------------------------------------------------------
oof_xgb = np.zeros((len(X), 3))
oof_cbt = np.zeros((len(X), 3))

print("\nTraining 5-Fold OOF...")
for fold in range(5):
    print(f"  Fold {fold}...", end=" ")
    tr_mask = train_bags['fold'] != fold
    val_mask = train_bags['fold'] == fold

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    w_tr = sample_wts[tr_mask.values]

    # XGBoost — OUR proven params
    m_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        max_depth=5, learning_rate=0.05, n_estimators=2000,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3, reg_alpha=0.5, min_child_weight=3,
        random_state=SEED, n_jobs=4, verbosity=0,
        eval_metric='mlogloss', early_stopping_rounds=100
    )
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_mask.values] = m_xgb.predict_proba(X_val)

    # CatBoost — OUR proven params
    m_cbt = CatBoostClassifier(
        loss_function='MultiClass', eval_metric='TotalF1',
        class_weights=class_weights,
        depth=6, learning_rate=0.05, iterations=2000,
        early_stopping_rounds=100, l2_leaf_reg=6,
        random_seed=SEED, verbose=False
    )
    m_cbt.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cbt[val_mask.values] = m_cbt.predict_proba(X_val)

    fx = f1_score(y_val, oof_xgb[val_mask.values].argmax(1), average='macro')
    fc = f1_score(y_val, oof_cbt[val_mask.values].argmax(1), average='macro')
    print(f"XGB={fx:.4f} CBT={fc:.4f}")

f1_xgb = f1_score(y, oof_xgb.argmax(1), average='macro')
f1_cbt = f1_score(y, oof_cbt.argmax(1), average='macro')
print(f"\nOOF  XGB={f1_xgb:.4f}  CBT={f1_cbt:.4f}")

# ------------------------------------------------------------------
# 3. ENSEMBLE SEARCH (finer grid: 0.05 steps)
# ------------------------------------------------------------------
print("\nSearching ensemble weights...")
best_ens_f1 = 0.0
best_wx = 1.0

for wx in np.arange(0.5, 1.01, 0.05):
    wc = 1.0 - wx
    blend = wx * oof_xgb + wc * oof_cbt
    sc = f1_score(y, blend.argmax(1), average='macro')
    if sc > best_ens_f1:
        best_ens_f1, best_wx = sc, wx
    print(f"  XGB={wx:.2f} Cat={wc:.2f} -> F1={sc:.4f}")

best_wc = 1.0 - best_wx
print(f"\nBest ensemble: XGB={best_wx:.2f} Cat={best_wc:.2f}  F1={best_ens_f1:.4f}")

oof_ens = best_wx * oof_xgb + best_wc * oof_cbt

# ------------------------------------------------------------------
# 4. ADDITIVE THRESHOLD TUNING — WIDE SEARCH
# ------------------------------------------------------------------
print("\nSearching additive thresholds (±0.15)...")
base_f1 = f1_score(y, oof_ens.argmax(1), average='macro')
best_tuned = base_f1
best_d0, best_d1, best_d2 = 0.0, 0.0, 0.0

deltas = np.round(np.arange(-0.15, 0.16, 0.02), 2)
count = 0

for d0 in deltas:
    for d1 in deltas:
        for d2 in deltas:
            adj = oof_ens.copy()
            adj[:, 0] += d0
            adj[:, 1] += d1
            adj[:, 2] += d2
            sc = f1_score(y, adj.argmax(1), average='macro')
            if sc > best_tuned:
                best_tuned = sc
                best_d0, best_d1, best_d2 = d0, d1, d2
            count += 1

print(f"Searched {count} combos.")
print(f"Best deltas: lower={best_d0:+.2f} middle={best_d1:+.2f} upper={best_d2:+.2f}")
print(f"Tuned Macro F1: {best_tuned:.4f}  (gain: +{best_tuned - base_f1:.4f})")

# ------------------------------------------------------------------
# 5. FINAL REPORT
# ------------------------------------------------------------------
adj = oof_ens.copy()
adj[:, 0] += best_d0
adj[:, 1] += best_d1
adj[:, 2] += best_d2
oof_preds = adj.argmax(1)

print("\n" + classification_report(y, oof_preds,
                                   target_names=['lower', 'middle', 'upper'], digits=4))
per_class = f1_score(y, oof_preds, average=None)
print(f"lower={per_class[0]:.4f}  middle={per_class[1]:.4f} (tie-break)  upper={per_class[2]:.4f}")

# ------------------------------------------------------------------
# 6. FINAL TRAIN + SUBMIT
# ------------------------------------------------------------------
print("\nTraining final models...")

final_xgb = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3,
    max_depth=5, learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=3, reg_alpha=0.5, min_child_weight=3,
    random_state=SEED, n_jobs=4, verbosity=0
)
final_xgb.fit(X, y, sample_weight=sample_wts, verbose=False)

final_cbt = CatBoostClassifier(
    loss_function='MultiClass', class_weights=class_weights,
    depth=6, learning_rate=0.05, iterations=800,
    l2_leaf_reg=6, random_seed=SEED, verbose=False
)
final_cbt.fit(X, y, verbose=False)

# Test prediction with tuned thresholds
test_proba = best_wx * final_xgb.predict_proba(X_test) + best_wc * final_cbt.predict_proba(X_test)
test_proba[:, 0] += best_d0
test_proba[:, 1] += best_d1
test_proba[:, 2] += best_d2
test_preds = test_proba.argmax(1)

submission = pd.DataFrame({'bag_id': test_bags['bag_id'].values, 'label': test_preds})
submission.to_csv('submission.csv', index=False)

print("\n" + "=" * 70)
print("FINAL SUBMISSION GENERATED")
print("=" * 70)
print(f"File: submission.csv")
print(f"Expected Macro F1: {best_tuned:.4f}")
print(f"Ensemble: XGB={best_wx:.2f} Cat={best_wc:.2f}")
print(f"Thresholds: lower={best_d0:+.2f} middle={best_d1:+.2f} upper={best_d2:+.2f}")
print("\n>>> UPLOAD TO KAGGLE <<<")
