"""
Step 3: Create Stratified Group K-Fold Splits
==============================================
WHY THIS FILE:
  - We must validate on BAGS, not random rows. Otherwise the same household
    appears in both train and validation, causing fake CV scores.
  - Stratification ensures every fold has the same % of lower/middle/upper.
  - 5-fold gives us reliable Macro F1 estimates AND out-of-fold predictions.

WHY RUN IT:
  - Step 4 reads folds.csv to know which bags are train vs validation in each fold.
  - You can inspect folds.csv to verify label balance per fold.

COMMAND TO RUN:
  python step3_create_folds.py

OUTPUT:
  data/folds.csv  (columns: bag_id, fold)
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

print("=" * 60)
print("STEP 3: Creating stratified group folds...")
print("=" * 60)

# Load bag-level data
train_bags = pd.read_csv('data/train_bags.csv')

# Get unique bag_ids with their labels
bags = train_bags[['bag_id', 'label']].drop_duplicates().reset_index(drop=True)
print(f"Total bags to split: {len(bags)}")
print(f"Label distribution:\n{bags['label'].value_counts().sort_index()}")

# Stratified K-Fold on bag_id level
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
bags['fold'] = -1

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(bags, bags['label'])):
    bags.iloc[val_idx, bags.columns.get_loc('fold')] = fold_idx
    fold_labels = bags.iloc[val_idx]['label'].value_counts().sort_index()
    print(f"\nFold {fold_idx} validation bags: {len(val_idx)}")
    print(fold_labels.to_dict())

# Save
bags[['bag_id', 'fold']].to_csv('data/folds.csv', index=False)

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"Saved: data/folds.csv")
print("\nNext: Run step4_cv_training.py")
