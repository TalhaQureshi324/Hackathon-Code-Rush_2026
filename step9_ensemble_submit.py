"""
Step 9: Ensemble Search + Final Submission
===========================================
WHY THIS FILE:
  - XGBoost OOF = 0.7086, CatBoost OOF = 0.6827.
  - We grid-search ensemble weights (0.0 to 1.0) to find the best mix.
  - Then trains final models on 100% train data and generates submission.csv.

WHY RUN IT:
  - This creates your Kaggle submission.
  - If ensemble beats pure XGBoost, you submit the ensemble.
  - If not, you submit pure XGBoost.

COMMAND TO RUN:
  python step9_ensemble_submit.py

OUTPUTS:
  submission.csv
  models/final_catboost.cbm
  models/final_xgboost.json
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

print("=" * 70)
print("STEP 9: Ensemble Search + Final Submission")
print("=" * 70)

# ------------------------------------------------------------------
# 1. LOAD CLEAN DATA
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
# 2. CATBOOST OOF (clean features, best weights)
# ------------------------------------------------------------------
print("\nRunning CatBoost OOF (silent)...")
cat_oof_proba = np.zeros((len(train_bags), 3))

for fold in range(5):
    tr_mask = train_bags['fold'] != fold
    val_mask = train_bags['fold'] == fold

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
        verbose=False
    )
    model.fit(X[tr_mask], y[tr_mask],
              eval_set=(X[val_mask], y[val_mask]),
              verbose=False)
    cat_oof_proba[val_mask] = model.predict_proba(X[val_mask])

cat_preds = cat_oof_proba.argmax(axis=1)
cat_f1 = f1_score(y, cat_preds, average='macro')
print(f"CatBoost OOF Macro F1: {cat_f1:.4f}")

# ------------------------------------------------------------------
# 3. LOAD XGBOOST OOF
# ------------------------------------------------------------------
xgb_oof_proba = np.load('data/oof_proba_xgb.npy')
xgb_preds = xgb_oof_proba.argmax(axis=1)
xgb_f1 = f1_score(y, xgb_preds, average='macro')
print(f"XGBoost OOF Macro F1:  {xgb_f1:.4f}")

# ------------------------------------------------------------------
# 4. ENSEMBLE WEIGHT SEARCH
# ------------------------------------------------------------------
print("\nSearching ensemble weights...")
best_f1 = 0
best_w = 0.0

for w in np.arange(0.0, 1.05, 0.05):
    ensemble_proba = w * xgb_oof_proba + (1 - w) * cat_oof_proba
    preds = ensemble_proba.argmax(axis=1)
    f1 = f1_score(y, preds, average='macro')
    marker = " <-- BEST" if f1 > best_f1 else ""
    print(f"  w={w:.2f} (XGB) / {1-w:.2f} (Cat)  ->  Macro F1: {f1:.4f}{marker}")
    if f1 > best_f1:
        best_f1 = f1
        best_w = w

print(f"\n>>> BEST ENSEMBLE: {best_w:.2f} XGBoost + {1-best_w:.2f} CatBoost <<<")
print(f">>> OOF Macro F1: {best_f1:.4f} <<<")

# ------------------------------------------------------------------
# 5. TRAIN FINAL MODELS ON 100% DATA
# ------------------------------------------------------------------
print("\nTraining final CatBoost on all data...")
final_cat = CatBoostClassifier(
    loss_function='MultiClass',
    class_weights=class_weights,
    depth=6,
    learning_rate=0.05,
    iterations=800,  # conservative, based on early stopping from CV
    l2_leaf_reg=6,
    random_seed=42,
    verbose=200
)
final_cat.fit(X, y, verbose=200)
final_cat.save_model('models/final_catboost.cbm')
print("Saved: models/final_catboost.cbm")

print("\nTraining final XGBoost on all data...")
sample_weight = np.array([class_weights[int(v)] for v in y])
final_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=500,  # based on CV best iterations ~400-500
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3,
    reg_alpha=0.5,
    min_child_weight=3,
    random_state=42,
    n_jobs=4,
    verbosity=0
)
final_xgb.fit(X, y, sample_weight=sample_weight)
final_xgb.save_model('models/final_xgboost.json')
print("Saved: models/final_xgboost.json")

# ------------------------------------------------------------------
# 6. PREDICT TEST SET
# ------------------------------------------------------------------
print("\nPredicting test set...")
test_bags = pd.read_csv('data/test_bags.csv')
test_bags = test_bags.drop(columns=noise_cols, errors='ignore')
X_test = test_bags[feature_cols]

cat_test_proba = final_cat.predict_proba(X_test)
xgb_test_proba = final_xgb.predict_proba(X_test)

ensemble_test_proba = best_w * xgb_test_proba + (1 - best_w) * cat_test_proba
test_preds = ensemble_test_proba.argmax(axis=1)

# ------------------------------------------------------------------
# 7. SAVE SUBMISSION
# ------------------------------------------------------------------
submission = pd.DataFrame({
    'bag_id': test_bags['bag_id'].values,
    'label': test_preds
})
submission.to_csv('submission.csv', index=False)

print("\n" + "=" * 70)
print("SUBMISSION GENERATED")
print("=" * 70)
print(f"File: submission.csv")
print(f"Shape: {submission.shape}")
print(f"Label distribution:")
print(submission['label'].value_counts().sort_index())
print(f"\nModel used: {best_w:.2f} XGBoost + {1-best_w:.2f} CatBoost")
print(f"Expected Macro F1 (OOF): {best_f1:.4f}")
print("\n>>> Ready to upload to Kaggle! <<<")
