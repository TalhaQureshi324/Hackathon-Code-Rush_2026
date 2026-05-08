"""
Step 8: XGBoost Cross-Validation
=================================
WHY THIS FILE:
  - Tier 2 failed (0.6063). We go back to clean bag features.
  - XGBoost is a strong alternative to CatBoost. It optimizes splits differently.
  - If XGBoost OOF is close to CatBoost (0.6827), we ensemble them for a boost.

WHY RUN IT:
  - Gives us a second high-quality model.
  - Ensembling CatBoost + XGBoost is a proven hackathon winning move.

COMMAND TO RUN:
  python step8_xgboost_cv.py

NOTE:
  If XGBoost is not installed:
    pip install xgboost

OUTPUTS:
  models/xgb_fold0.json ... fold4.json
  data/oof_predictions_xgb.csv
  data/oof_proba_xgb.npy
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# Check xgboost
try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed.")
    print("Run:  pip install xgboost")
    sys.exit(1)

print("=" * 70)
print("STEP 8: XGBoost 5-Fold CV")
print("=" * 70)

# ------------------------------------------------------------------
# Load clean bag data (same as Step 5, no Tier 2)
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags.csv')
folds = pd.read_csv('data/folds.csv')
train_bags = train_bags.merge(folds, on='bag_id', how='left')

# Drop noise features (same as Step 5)
noise_cols = [c for c in train_bags.columns if 'survey_duration_mins' in c]
if noise_cols:
    print(f"Dropping noise: {noise_cols}")
    train_bags = train_bags.drop(columns=noise_cols)

feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label', 'fold']]
X = train_bags[feature_cols]
y = train_bags['label']

print(f"Features: {len(feature_cols)}")
print(f"Bags: {len(train_bags)}")

# Best weights from Step 5
class_weights = {0: 1.1, 1: 1.4, 2: 1.0}

# ------------------------------------------------------------------
# 5-Fold CV
# ------------------------------------------------------------------
oof_preds = np.zeros(len(train_bags), dtype=int)
oof_proba = np.zeros((len(train_bags), 3))

for fold in range(5):
    print(f"\n{'='*70}")
    print(f"Fold {fold}")
    print(f"{'='*70}")

    tr_mask = train_bags['fold'] != fold
    val_mask = train_bags['fold'] == fold

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Sample weights for class imbalance (middle class boost)
    sample_weight = np.array([class_weights[int(v)] for v in y_tr])

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3,
        reg_alpha=0.5,
        min_child_weight=3,
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=100,
        n_jobs=4,
        verbosity=0
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val).astype(int)
    proba = model.predict_proba(X_val)

    oof_preds[val_mask] = preds
    oof_proba[val_mask] = proba

    fold_f1 = f1_score(y_val, preds, average='macro')
    print(f"--> Fold {fold} Macro F1: {fold_f1:.4f}  (best_iter={model.best_iteration})")
    model.save_model(f'models/xgb_fold{fold}.json')
    print(f"Saved: models/xgb_fold{fold}.json")

# ------------------------------------------------------------------
# Overall OOF
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("XGBOOST CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

overall_f1 = f1_score(y, oof_preds, average='macro')
print(f"\n>>> OOF Macro F1: {overall_f1:.4f} <<<")

print("\nPer-class F1:")
print(classification_report(y, oof_preds,
                            target_names=['lower (0)', 'middle (1)', 'upper (2)'],
                            digits=4))

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
oof_df = train_bags[['bag_id', 'label']].copy()
oof_df['pred'] = oof_preds
oof_df.to_csv('data/oof_predictions_xgb.csv', index=False)
np.save('data/oof_proba_xgb.npy', oof_proba)

print(f"\nSaved: data/oof_predictions_xgb.csv")
print(f"Saved: data/oof_proba_xgb.npy")
print("\nNext: Run step9_ensemble.py")
