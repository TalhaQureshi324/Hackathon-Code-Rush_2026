"""
Step 11: Supercharged Hybrid — Your Features + Partner's Methods
=================================================================
WHY THIS FILE:
  - Your features are richer (quantiles, interactions, household composition).
  - Partner's methods are better tuned (XGB params, LightGBM, additive threshold tuning, class weights).
  - This combines both to create the strongest possible model.

WHY RUN IT:
  - This is your FINAL winning submission.
  - Expected OOF: 0.725–0.735.

COMMAND TO RUN:
  python step11_supercharged.py

OUTPUT:
  submission.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------
# Check dependencies
# ------------------------------------------------------------------
try:
    import xgboost as xgb
except ImportError:
    print("pip install xgboost"); sys.exit(1)

try:
    import lightgbm as lgb
except ImportError:
    print("pip install lightgbm"); sys.exit(1)

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("pip install catboost"); sys.exit(1)

print("=" * 70)
print("STEP 11: Supercharged Hybrid Model")
print("=" * 70)

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. LOAD YOUR RICH BAG FEATURES
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags.csv')
test_bags = pd.read_csv('data/test_bags.csv')
folds = pd.read_csv('data/folds.csv')

# Drop noise
train_bags = train_bags.merge(folds, on='bag_id', how='left')
noise_cols = [c for c in train_bags.columns if 'survey_duration_mins' in c]
if noise_cols:
    train_bags = train_bags.drop(columns=noise_cols)
    test_bags = test_bags.drop(columns=noise_cols)

feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label', 'fold']]
X = train_bags[feature_cols].values.astype(np.float32)
y = train_bags['label'].values
X_test = test_bags[feature_cols].values.astype(np.float32)

print(f"Features: {len(feature_cols)}")
print(f"Train: {X.shape}  |  Test: {X_test.shape}")

# ------------------------------------------------------------------
# 2. CLASS WEIGHTS (Partner's formula — heavily boosts lower class)
# ------------------------------------------------------------------
class_counts = np.bincount(y)
class_wts = len(y) / (3 * class_counts)
class_wts[0] *= 1.5
class_wts = class_wts / class_wts.mean()
sample_wts = np.array([class_wts[yi] for yi in y])
print(f"Class weights: lower={class_wts[0]:.3f}  middle={class_wts[1]:.3f}  upper={class_wts[2]:.3f}")

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------------------------------------------------
# 3. MODEL PARAMETERS (Partner's tuned settings)
# ------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.80, colsample_bytree=0.75, min_child_weight=3,
    gamma=0.05, reg_alpha=0.3, reg_lambda=1.5,
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    early_stopping_rounds=100, use_label_encoder=False,
    random_state=SEED, n_jobs=-1, verbosity=0,
)

LGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    max_depth=-1, min_child_samples=15,
    subsample=0.80, colsample_bytree=0.75,
    reg_alpha=0.3, reg_lambda=1.5,
    class_weight="balanced",
    objective="multiclass", num_class=3, metric="multi_logloss",
    random_state=SEED, n_jobs=-1, verbose=-1,
)

CBT_PARAMS = dict(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
    bootstrap_type="Bernoulli",
    subsample=0.80,
    rsm=0.75,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    class_weights=class_wts.tolist(),
    early_stopping_rounds=100,
    random_seed=SEED,
    verbose=False,
)

# ------------------------------------------------------------------
# 4. OOF TRAINING (3 models)
# ------------------------------------------------------------------
oof_xgb = np.zeros((len(X), 3))
oof_lgb = np.zeros((len(X), 3))
oof_cbt = np.zeros((len(X), 3))
test_xgb = np.zeros((len(X_test), 3))
test_lgb = np.zeros((len(X_test), 3))
test_cbt = np.zeros((len(X_test), 3))

print("\nTraining 5-Fold OOF for XGBoost + LightGBM + CatBoost...")

for fold, (tr_idx, val_idx) in enumerate(SKF.split(X, y)):
    Xtr, Xval = X[tr_idx], X[val_idx]
    ytr, yval = y[tr_idx], y[val_idx]
    wtr = sample_wts[tr_idx]

    # XGBoost
    m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    m_xgb.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict_proba(Xval)
    test_xgb += m_xgb.predict_proba(X_test) / SKF.n_splits

    # LightGBM
    m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    try:
        m_lgb.fit(Xtr, ytr, sample_weight=wtr,
                  eval_set=[(Xval, yval)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    except TypeError:
        # Fallback for older LightGBM versions
        m_lgb.fit(Xtr, ytr, sample_weight=wtr,
                  eval_set=[(Xval, yval)],
                  early_stopping_rounds=100, verbose=False)
    oof_lgb[val_idx] = m_lgb.predict_proba(Xval)
    test_lgb += m_lgb.predict_proba(X_test) / SKF.n_splits

    # CatBoost
    m_cbt = CatBoostClassifier(**CBT_PARAMS)
    m_cbt.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_cbt[val_idx] = m_cbt.predict_proba(Xval)
    test_cbt += m_cbt.predict_proba(X_test) / SKF.n_splits

    x_f1 = f1_score(yval, oof_xgb[val_idx].argmax(1), average="macro")
    l_f1 = f1_score(yval, oof_lgb[val_idx].argmax(1), average="macro")
    c_f1 = f1_score(yval, oof_cbt[val_idx].argmax(1), average="macro")
    print(f"  Fold {fold+1}  XGB={x_f1:.4f}  LGB={l_f1:.4f}  CBT={c_f1:.4f}")

f1_xgb = f1_score(y, oof_xgb.argmax(1), average="macro")
f1_lgb = f1_score(y, oof_lgb.argmax(1), average="macro")
f1_cbt = f1_score(y, oof_cbt.argmax(1), average="macro")
print(f"\nOOF  XGB={f1_xgb:.4f}  LGB={f1_lgb:.4f}  CBT={f1_cbt:.4f}")

# ------------------------------------------------------------------
# 5. 3-MODEL ENSEMBLE SEARCH
# ------------------------------------------------------------------
print("\nSearching ensemble weights...")
best_ens_f1 = 0.0
best_wx, best_wl, best_wc = 1.0, 0.0, 0.0

for wx in np.arange(0.0, 1.05, 0.1):
    for wl in np.arange(0.0, 1.05 - wx, 0.1):
        wc = round(1.0 - wx - wl, 2)
        if wc < 0:
            continue
        blend = wx * oof_xgb + wl * oof_lgb + wc * oof_cbt
        sc = f1_score(y, blend.argmax(1), average="macro")
        if sc > best_ens_f1:
            best_ens_f1, best_wx, best_wl, best_wc = sc, wx, wl, wc

print(f"Best ensemble: XGB={best_wx:.1f}  LGB={best_wl:.1f}  CBT={best_wc:.1f}  F1={best_ens_f1:.4f}")

oof_ensemble = best_wx * oof_xgb + best_wl * oof_lgb + best_wc * oof_cbt
test_ensemble = best_wx * test_xgb + best_wl * test_lgb + best_wc * test_cbt

# ------------------------------------------------------------------
# 6. ADDITIVE THRESHOLD TUNING (Partner's secret weapon)
# ------------------------------------------------------------------
print("\nSearching additive thresholds...")
base_f1 = f1_score(y, oof_ensemble.argmax(1), average="macro")
best_tuned = base_f1
best_d0, best_d1, best_d2 = 0.0, 0.0, 0.0

deltas = np.round(np.arange(-0.10, 0.11, 0.02), 2)

for d0 in deltas:
    for d1 in deltas:
        for d2 in deltas:
            adj = oof_ensemble.copy()
            adj[:, 0] += d0
            adj[:, 1] += d1
            adj[:, 2] += d2
            sc = f1_score(y, adj.argmax(1), average="macro")
            if sc > best_tuned:
                best_tuned = sc
                best_d0, best_d1, best_d2 = d0, d1, d2

print(f"Best deltas: lower={best_d0:+.2f}  middle={best_d1:+.2f}  upper={best_d2:+.2f}")
print(f"Tuned Macro F1: {best_tuned:.4f}  (gain: +{best_tuned - base_f1:.4f})")

# ------------------------------------------------------------------
# 7. FINAL REPORT
# ------------------------------------------------------------------
adj = oof_ensemble.copy()
adj[:, 0] += best_d0
adj[:, 1] += best_d1
adj[:, 2] += best_d2
oof_preds = adj.argmax(1)

print("\n" + classification_report(y, oof_preds, target_names=["lower","middle","upper"], digits=4))
per_class = f1_score(y, oof_preds, average=None)
print(f"lower={per_class[0]:.4f}  middle={per_class[1]:.4f} (tie-break)  upper={per_class[2]:.4f}")

# ------------------------------------------------------------------
# 8. TRAIN FINAL MODELS + SUBMIT
# ------------------------------------------------------------------
print("\nTraining final models on 100% data...")

final_xgb = xgb.XGBClassifier(**{k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"})
final_xgb.fit(X, y, sample_weight=sample_wts, verbose=False)

try:
    final_lgb = lgb.LGBMClassifier(**{k: v for k, v in LGB_PARAMS.items() if k != "metric"})
except:
    final_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
final_lgb.fit(X, y, sample_weight=sample_wts)

final_cbt = CatBoostClassifier(**{**CBT_PARAMS, "iterations": 800, "early_stopping_rounds": None})
final_cbt.fit(X, y, verbose=False)

# Predict test
test_adj = (best_wx * final_xgb.predict_proba(X_test) +
            best_wl * final_lgb.predict_proba(X_test) +
            best_wc * final_cbt.predict_proba(X_test))
test_adj[:, 0] += best_d0
test_adj[:, 1] += best_d1
test_adj[:, 2] += best_d2
test_preds = test_adj.argmax(1)

submission = pd.DataFrame({"bag_id": test_bags["bag_id"].values, "label": test_preds})
submission.to_csv("submission.csv", index=False)

print("\n" + "=" * 70)
print("FINAL SUBMISSION GENERATED")
print("=" * 70)
print(f"File: submission.csv  |  Rows: {len(submission)}")
print(f"Label dist: {dict(submission['label'].value_counts().sort_index())}")
print(f"Expected Macro F1: {best_tuned:.4f}")
print(f"Ensemble: XGB={best_wx:.1f} LGB={best_wl:.1f} CBT={best_wc:.1f}")
print(f"Thresholds: lower={best_d0:+.2f} middle={best_d1:+.2f} upper={best_d2:+.2f}")
print("\n>>> UPLOAD TO KAGGLE NOW <<<")
