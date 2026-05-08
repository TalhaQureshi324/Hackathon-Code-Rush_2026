"""
Step 5: Class Weight & Noise Optimization
==========================================
WHY THIS FILE:
  - Our baseline (0.6527) is losing on the middle class (F1=0.59).
  - 'Balanced' weights actually DOWN-weight middle because it has the most samples.
  - We need to UP-weight middle class to win the tie-breaker and boost Macro F1.
  - Also drops survey_duration_mins (0.007 correlation = pure noise).
  - Tries depth=6 to capture more interactions without excessive overfitting.

WHY RUN IT:
  - Finds the exact class weights that maximize YOUR OOF Macro F1.
  - Takes ~3-5 minutes but can jump you +0.02 to +0.05 instantly.

COMMAND TO RUN:
  python step5_optimize.py

OUTPUT:
  Prints a ranked table of configs.
  Saves the best config to data/best_weights.json for later steps.
"""

import ast
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

print("=" * 70)
print("STEP 5: Grid Search for Best Class Weights + Noise Cleanup")
print("=" * 70)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
train_bags = pd.read_csv('data/train_bags.csv')
folds = pd.read_csv('data/folds.csv')
train_bags = train_bags.merge(folds, on='bag_id', how='left')

feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label', 'fold']]

# ------------------------------------------------------------------
# Drop obvious noise features
# ------------------------------------------------------------------
noise_cols = [c for c in feature_cols if 'survey_duration_mins' in c]
if noise_cols:
    print(f"Dropping {len(noise_cols)} noise features: {noise_cols[:3]}...")
    feature_cols = [c for c in feature_cols if c not in noise_cols]

X = train_bags[feature_cols]
y = train_bags['label']

print(f"Features after cleanup: {len(feature_cols)}")

# ------------------------------------------------------------------
# Weight configurations to test
# Format: {0: lower_weight, 1: middle_weight, 2: upper_weight}
# ------------------------------------------------------------------
weight_configs = [
    {"name": "balanced_sklearn", "weights": None},  # computed per fold
    {"name": "no_weights",     "weights": {0: 1.0, 1: 1.0, 2: 1.0}},
    {"name": "middle_1.2",     "weights": {0: 1.0, 1: 1.2, 2: 1.0}},
    {"name": "middle_1.3",     "weights": {0: 1.0, 1: 1.3, 2: 1.0}},
    {"name": "middle_1.4",     "weights": {0: 1.0, 1: 1.4, 2: 1.0}},
    {"name": "middle_1.5",     "weights": {0: 1.0, 1: 1.5, 2: 1.0}},
    {"name": "middle_1.6",     "weights": {0: 1.0, 1: 1.6, 2: 1.0}},
    {"name": "middle_1.5_lower_1.2", "weights": {0: 1.2, 1: 1.5, 2: 1.0}},
    {"name": "middle_1.5_upper_1.2", "weights": {0: 1.0, 1: 1.5, 2: 1.2}},
    {"name": "middle_1.3_lower_1.1_upper_0.9", "weights": {0: 1.1, 1: 1.3, 2: 0.9}},
    {"name": "lower_1.3_middle_1.3_upper_0.8", "weights": {0: 1.3, 1: 1.3, 2: 0.8}},
    {"name": "lower_1.2_middle_1.4_upper_0.9", "weights": {0: 1.2, 1: 1.4, 2: 0.9}},
    {"name": "middle_1.4_lower_1.1_upper_1.0", "weights": {0: 1.1, 1: 1.4, 2: 1.0}},
    {"name": "middle_2.0",     "weights": {0: 1.0, 1: 2.0, 2: 1.0}},
    {"name": "middle_1.5_lower_0.9_upper_1.1", "weights": {0: 0.9, 1: 1.5, 2: 1.1}},
    {"name": "middle_1.3_lower_0.9_upper_1.0", "weights": {0: 0.9, 1: 1.3, 2: 1.0}},
]

results = []

for cfg in weight_configs:
    print(f"\n--- Testing: {cfg['name']} ---")
    oof_preds = np.zeros(len(train_bags), dtype=int)

    for fold in range(5):
        tr_mask = train_bags['fold'] != fold
        val_mask = train_bags['fold'] == fold
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Determine weights
        if cfg['weights'] is None:
            classes = np.array([0, 1, 2])
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
            class_weights = dict(zip(classes, cw))
        else:
            class_weights = cfg['weights']

        model = CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='TotalF1',
            class_weights=class_weights,
            depth=6,                     # slightly deeper than baseline
            learning_rate=0.05,
            iterations=2000,
            early_stopping_rounds=100,
            l2_leaf_reg=6,               # stronger regularization to compensate for depth=6
            random_seed=42,
            verbose=False                # silent for grid search
        )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val).ravel().astype(int)
        oof_preds[val_mask] = preds

    macro_f1 = f1_score(y, oof_preds, average='macro')
    f1_per_class = f1_score(y, oof_preds, average=None)

    results.append({
        'name': cfg['name'],
        'macro_f1': macro_f1,
        'f1_lower': f1_per_class[0],
        'f1_middle': f1_per_class[1],
        'f1_upper': f1_per_class[2],
        'weights': str(cfg['weights'])
    })

    print(f"Macro F1: {macro_f1:.4f}  |  Lower: {f1_per_class[0]:.4f}  Middle: {f1_per_class[1]:.4f}  Upper: {f1_per_class[2]:.4f}")

# ------------------------------------------------------------------
# Rank results
# ------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values('macro_f1', ascending=False)
print("\n" + "=" * 70)
print("RANKED RESULTS (Best First)")
print("=" * 70)
print(results_df[['name', 'macro_f1', 'f1_lower', 'f1_middle', 'f1_upper']].to_string(index=False))

best = results_df.iloc[0]
print(f"\n>>> BEST CONFIG: {best['name']} <<<")
print(f">>> Macro F1: {best['macro_f1']:.4f} <<<")
print(f">>> Weights: {best['weights']} <<<")

# Save best config
best_cfg = {
    'name': best['name'],
    'macro_f1': float(best['macro_f1']),
    'weights': ast.literal_eval(best['weights']) if best['weights'] != 'None' else None
}
with open('data/best_weights.json', 'w') as f:
    json.dump(best_cfg, f, indent=2)

print("\nSaved: data/best_weights.json")
print("\nNext: We use these weights in Step 6 (Tier 2 Meta-Features) or Step 7 (Final Training).")
