"""
Step 7: Final CV on Tier 2 Features
====================================
WHY THIS FILE:
  - Trains bag-level CatBoost on the TIER 2 dataset:
    (bag stats + row-probability meta-features + best class weights).
  - This tells us if Tier 2 meta-features actually boosted performance.
  - Expected: 0.71–0.74 Macro F1 if meta-features are informative.

WHY RUN IT:
  - This is your new baseline. If it beats 0.6827, Tier 2 worked.
  - Saves fold models for reproducibility.

COMMAND TO RUN:
  python step7_final_cv_tier2.py

OUTPUTS:
  models/catboost_t2_fold0.cbm ... fold4.cbm
  data/oof_predictions_tier2.csv
  data/oof_proba_tier2.npy
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, classification_report

print("=" * 70)
print("STEP 7: Final CV on Tier 2 Data")
print("=" * 70)

# ------------------------------------------------------------------
# Load Tier 2 bag data
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags_tier2.csv')
folds = pd.read_csv('data/folds.csv')
train_bags = train_bags.merge(folds, on='bag_id', how='left')

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

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        depth=6,
        learning_rate=0.05,
        iterations=2000,
        early_stopping_rounds=100,
        l2_leaf_reg=6,
        random_seed=42,
        verbose=100
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=100)

    preds = model.predict(X_val).ravel().astype(int)
    proba = model.predict_proba(X_val)

    oof_preds[val_mask] = preds
    oof_proba[val_mask] = proba

    fold_f1 = f1_score(y_val, preds, average='macro')
    print(f"\n--> Fold {fold} Macro F1: {fold_f1:.4f}")
    model.save_model(f'models/catboost_t2_fold{fold}.cbm')
    print(f"Saved: models/catboost_t2_fold{fold}.cbm")

# ------------------------------------------------------------------
# Overall OOF Evaluation
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("TIER 2 CROSS-VALIDATION RESULTS")
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
oof_df.to_csv('data/oof_predictions_tier2.csv', index=False)
np.save('data/oof_proba_tier2.npy', oof_proba)

print(f"\nSaved: data/oof_predictions_tier2.csv")
print(f"Saved: data/oof_proba_tier2.npy")
print("\nNext: Run step8_final_train.py")
