"""
Step 4: CatBoost Cross-Validation Training
============================================
WHY THIS FILE:
  - Trains 5 CatBoost models (one per fold) with early stopping.
  - Evaluates on Macro F1 (the competition metric) using out-of-fold (OOF) predictions.
  - Saves every fold model so we can reproduce predictions later.
  - OOF predictions give us an unbiased estimate of hidden-test performance.

WHY RUN IT:
  - This tells us our REAL score before submitting to Kaggle.
  - If OOF Macro F1 is 0.72, hidden test will likely be ~0.70-0.74.
  - We also save OOF probabilities for Tier 2 meta-stacking later.

COMMAND TO RUN:
  python step4_cv_training.py

NOTE:
  If CatBoost is not installed, run:
    pip install catboost

OUTPUTS:
  models/catboost_fold0.cbm ... fold4.cbm
  data/oof_predictions.csv
  data/oof_proba.npy
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------
# Check CatBoost
# ------------------------------------------------------------------
try:
    from catboost import CatBoostClassifier
except ImportError:
    print("ERROR: CatBoost not installed.")
    print("Run:  pip install catboost")
    sys.exit(1)

print("=" * 70)
print("STEP 4: CatBoost 5-Fold CV Training")
print("=" * 70)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags.csv')
folds = pd.read_csv('data/folds.csv')

# Merge fold assignments
train_bags = train_bags.merge(folds, on='bag_id', how='left')

# Feature columns (drop IDs, target, fold)
feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label', 'fold']]
X = train_bags[feature_cols]
y = train_bags['label']

print(f"Features: {len(feature_cols)}")
print(f"Bags: {len(train_bags)}")

# ------------------------------------------------------------------
# Containers for OOF results
# ------------------------------------------------------------------
oof_preds = np.zeros(len(train_bags), dtype=int)
oof_proba = np.zeros((len(train_bags), 3))

# ------------------------------------------------------------------
# Train 5 folds
# ------------------------------------------------------------------
for fold in range(5):
    print(f"\n{'='*70}")
    print(f"Fold {fold}")
    print(f"{'='*70}")

    tr_mask = train_bags['fold'] != fold
    val_mask = train_bags['fold'] == fold

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Compute balanced class weights for THIS fold's train set
    classes = np.array([0, 1, 2])
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
    class_weights = dict(zip(classes, cw))
    print(f"Class weights: {class_weights}")

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',        # Macro F1 under the hood for multiclass
        class_weights=class_weights,
        depth=5,
        learning_rate=0.05,
        iterations=2000,
        early_stopping_rounds=100,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=100
    )

    # Predictions
    preds = model.predict(X_val).ravel().astype(int)
    proba = model.predict_proba(X_val)

    oof_preds[val_mask] = preds
    oof_proba[val_mask] = proba

    fold_f1 = f1_score(y_val, preds, average='macro')
    print(f"\n--> Fold {fold} Macro F1: {fold_f1:.4f}")

    # Save model
    model.save_model(f'models/catboost_fold{fold}.cbm')
    print(f"Saved: models/catboost_fold{fold}.cbm")

# ------------------------------------------------------------------
# Overall OOF Evaluation
# ------------------------------------------------------------------
print(f"\n{'='*70}")
print("CROSS-VALIDATION RESULTS (Out-of-Fold)")
print(f"{'='*70}")

overall_f1 = f1_score(y, oof_preds, average='macro')
print(f"\n>>> OOF Macro F1: {overall_f1:.4f} <<<")

print("\nPer-class F1:")
print(classification_report(y, oof_preds,
                            target_names=['lower (0)', 'middle (1)', 'upper (2)'],
                            digits=4))

# ------------------------------------------------------------------
# Save OOF outputs
# ------------------------------------------------------------------
oof_df = train_bags[['bag_id', 'label']].copy()
oof_df['pred'] = oof_preds
oof_df.to_csv('data/oof_predictions.csv', index=False)
np.save('data/oof_proba.npy', oof_proba)

print(f"\nSaved: data/oof_predictions.csv")
print(f"Saved: data/oof_proba.npy  (shape: {oof_proba.shape})")
print("\nNext: Run step5_oof_evaluation.py")
