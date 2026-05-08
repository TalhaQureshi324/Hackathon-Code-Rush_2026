"""
Step 6: Tier 2 — Row-Level Meta-Features
=========================================
WHY THIS FILE:
  - Bag-level averages dilute individual signals (e.g., one high-earner in a family).
  - We train a row-level CatBoost where each PERSON is labeled with their BAG's class.
  - We aggregate row predicted probabilities per bag (mean, max, std).
  - These meta-features capture "at least one upper-class person" logic that raw stats miss.

WHY RUN IT:
  - This is the highest-impact upgrade remaining. Can add +0.03–0.06 Macro F1.
  - It directly addresses why middle-class bags are hard: mixed households have 
    ambiguous averages but clear individual-level patterns.

COMMAND TO RUN:
  python step6_tier2_meta.py

OUTPUTS:
  data/train_bags_tier2.csv   (original bag features + 9 meta-features + label)
  data/test_bags_tier2.csv    (original bag features + 9 meta-features)
  models/row_catboost_final.cbm
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

print("=" * 70)
print("STEP 6: Building Tier 2 Row-Level Meta-Features")
print("=" * 70)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
train_rows = pd.read_csv('data/train_rows_clean.csv')
test_rows = pd.read_csv('data/test_rows_clean.csv')
folds = pd.read_csv('data/folds.csv')

# Merge fold info into row-level data
train_rows = train_rows.merge(folds, on='bag_id', how='left')

# Row-level features (everything except bag_id and label)
row_feature_cols = [c for c in train_rows.columns if c not in ['bag_id', 'label', 'fold']]

# CatBoost categorical indices (string columns still present at row level)
cat_cols = ['relationship', 'native_country', 'race', 'education',
            'workclass', 'occupation', 'sex', 'education_tier', 'marital_status']
cat_features = [row_feature_cols.index(c) for c in cat_cols if c in row_feature_cols]

print(f"Row features: {len(row_feature_cols)}")
print(f"Categorical features for row model: {cat_features}")

# Best weights from Step 5
row_class_weights = {0: 1.1, 1: 1.4, 2: 1.0}

# ------------------------------------------------------------------
# 2. OOF ROW-LEVEL TRAINING (5 folds)
# ------------------------------------------------------------------
print("\nTraining row-level model (OOF)...")

oof_row_proba = np.zeros((len(train_rows), 3))

for fold in range(5):
    print(f"  Fold {fold}...", end=" ")

    tr_mask = train_rows['fold'] != fold
    val_mask = train_rows['fold'] == fold

    X_tr, y_tr = train_rows.loc[tr_mask, row_feature_cols], train_rows.loc[tr_mask, 'label']
    X_val, y_val = train_rows.loc[val_mask, row_feature_cols], train_rows.loc[val_mask, 'label']

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=row_class_weights,
        depth=5,
        learning_rate=0.05,
        iterations=1500,
        early_stopping_rounds=80,
        l2_leaf_reg=5,
        random_seed=42 + fold,
        verbose=False
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features, verbose=False)
    proba = model.predict_proba(X_val)
    oof_row_proba[val_mask.values] = proba

    fold_f1 = f1_score(y_val, model.predict(X_val).ravel(), average='macro')
    print(f"Row F1={fold_f1:.4f}")

# ------------------------------------------------------------------
# 3. AGGREGATE ROW PROBABILITIES TO BAG LEVEL
# ------------------------------------------------------------------
print("\nAggregating row probabilities per bag...")

row_probs = pd.DataFrame({
    'bag_id': train_rows['bag_id'].values,
    'prob_0': oof_row_proba[:, 0],
    'prob_1': oof_row_proba[:, 1],
    'prob_2': oof_row_proba[:, 2]
})

# Train bag meta-features
meta_train = row_probs.groupby('bag_id').agg({
    'prob_0': ['mean', 'max', 'std'],
    'prob_1': ['mean', 'max', 'std'],
    'prob_2': ['mean', 'max', 'std']
})
meta_train.columns = [f"{c}_{s}" for c, s in meta_train.columns]
meta_train = meta_train.fillna(0).reset_index()

# ------------------------------------------------------------------
# 4. FINAL ROW MODEL → PREDICT TEST ROWS
# ------------------------------------------------------------------
print("Training final row model on all train data...")

final_row_model = CatBoostClassifier(
    loss_function='MultiClass',
    class_weights=row_class_weights,
    depth=5,
    learning_rate=0.05,
    iterations=1000,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=200
)

final_row_model.fit(
    train_rows[row_feature_cols], train_rows['label'],
    cat_features=cat_features,
    verbose=200
)

final_row_model.save_model('models/row_catboost_final.cbm')
print("Saved: models/row_catboost_final.cbm")

# Predict test rows
test_row_proba = final_row_model.predict_proba(test_rows[row_feature_cols])

test_row_probs = pd.DataFrame({
    'bag_id': test_rows['bag_id'].values,
    'prob_0': test_row_proba[:, 0],
    'prob_1': test_row_proba[:, 1],
    'prob_2': test_row_proba[:, 2]
})

meta_test = test_row_probs.groupby('bag_id').agg({
    'prob_0': ['mean', 'max', 'std'],
    'prob_1': ['mean', 'max', 'std'],
    'prob_2': ['mean', 'max', 'std']
})
meta_test.columns = [f"{c}_{s}" for c, s in meta_test.columns]
meta_test = meta_test.fillna(0).reset_index()

# ------------------------------------------------------------------
# 5. MERGE META-FEATURES INTO BAG-LEVEL DATA
# ------------------------------------------------------------------
print("Merging meta-features into bag-level data...")

train_bags = pd.read_csv('data/train_bags.csv')
test_bags = pd.read_csv('data/test_bags.csv')

# Drop old noise features if present (ensure consistency)
noise_cols = [c for c in train_bags.columns if 'survey_duration_mins' in c]
if noise_cols:
    train_bags = train_bags.drop(columns=noise_cols)
    test_bags = test_bags.drop(columns=noise_cols)

# Merge meta
train_bags_t2 = train_bags.merge(meta_train, on='bag_id', how='left')
test_bags_t2 = test_bags.merge(meta_test, on='bag_id', how='left')

# Save
train_bags_t2.to_csv('data/train_bags_tier2.csv', index=False)
test_bags_t2.to_csv('data/test_bags_tier2.csv', index=False)

print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print("=" * 70)
print(f"Train bags + meta: {train_bags_t2.shape}")
print(f"Test bags + meta:  {test_bags_t2.shape}")
print(f"New meta-features: {list(meta_train.columns[1:])}")
print("\nNext: Run step7_final_cv_tier2.py")
