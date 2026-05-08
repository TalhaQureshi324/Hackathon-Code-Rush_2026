"""
Coderush 2026 — ML Module: Economic Class Classification
Final Submission Notebook (Single File)
============================================================
Bag-level multi-class classification using XGBoost + CatBoost ensemble
with multiplicative threshold tuning optimized for Macro F1.

Run time: ~3-4 minutes on Kaggle CPU.
Outputs:
  /kaggle/working/submission.csv
  /kaggle/working/final_xgb.json
  /kaggle/working/final_catboost.cbm
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

# =============================================================================
# 1. PATH RESOLUTION (Kaggle vs Local)
# =============================================================================
_PATH_CANDIDATES = [
    "/kaggle/input/competitions/code-rush-26-ml-module/",
    "/kaggle/input/code-rush-26-ml-module/",
    "/kaggle/input/coderush-26-ml/",
    "",
]
TRAIN_PATH, TEST_PATH = None, None
for p in _PATH_CANDIDATES:
    t = os.path.join(p, "Coderush-26-ML-Train.csv")
    e = os.path.join(p, "Coderush-26-ML-test.csv")
    if os.path.exists(t):
        TRAIN_PATH, TEST_PATH = t, e
        break
if TRAIN_PATH is None:
    raise FileNotFoundError("Could not find dataset. Checked: " + str(_PATH_CANDIDATES))

# Output directory
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

print(f"Using train: {TRAIN_PATH}")
print(f"Using test:  {TEST_PATH}")

# =============================================================================
# 2. LOAD RAW DATA
# =============================================================================
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
print(f"Raw train: {train_raw.shape}  |  Raw test: {test_raw.shape}")

# =============================================================================
# 3. PREPROCESSING (Row-Level)
# =============================================================================
DROP_COLS = [
    'survey_year', 'currency_code', 'poverty_line_usd', 'processing_flag',
    'interview_mode', 'capital_activity_flag', 'interviewer_id', 'person_idx'
]
LABEL_MAP = {'lower': 0, 'middle': 1, 'upper': 2}

def preprocess(df, is_train=False):
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    
    # Age
    df['age'] = 1994 - df['year_of_birth']
    df = df.drop(columns=['year_of_birth'])
    
    # Log transforms
    df['log_capital_gain'] = np.log1p(df['capital_gain'])
    df['log_capital_loss'] = np.log1p(df['capital_loss'])
    
    # Interactions
    df['hours_x_education'] = df['hours_per_week'] * df['education_num']
    df['age_x_education'] = df['age'] * df['education_num']
    
    # Binary indicators
    df['is_married'] = (df['marital_status'] == 'Married-civ-spouse').astype(int)
    df['is_higher_edu'] = (df['education_num'] >= 13).astype(int)
    df['is_full_time'] = (df['hours_per_week'] >= 40).astype(int)
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'] > 0).astype(int)
    df['is_exec_managerial'] = (df['occupation'] == 'Exec-managerial').astype(int)
    df['is_prof_specialty'] = (df['occupation'] == 'Prof-specialty').astype(int)
    df['is_self_employed'] = df['workclass'].isin(['Self-emp-inc', 'Self-emp-not-inc']).astype(int)
    df['is_govt'] = df['workclass'].isin(['Federal-gov', 'Local-gov', 'State-gov']).astype(int)
    df['is_child'] = (df['relationship'] == 'Own-child').astype(int)
    df['is_husband'] = (df['relationship'] == 'Husband').astype(int)
    df['is_wife'] = (df['relationship'] == 'Wife').astype(int)
    
    if is_train:
        df['label'] = df['label'].map(LABEL_MAP)
    return df

train = preprocess(train_raw.copy(), is_train=True)
test = preprocess(test_raw.copy(), is_train=False)

# =============================================================================
# 4. BAG-LEVEL AGGREGATION
# =============================================================================
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)

def create_bag_features(df, is_train=False):
    gb = df.groupby('bag_id')
    
    # Numeric columns to aggregate
    numeric_cols = [
        'bag_size', 'education_num', 'capital_gain', 'capital_loss',
        'hours_per_week', 'net_capital_asset', 'is_adult_flag', 'annual_hours_est',
        'age', 'log_capital_gain', 'log_capital_loss',
        'hours_x_education', 'age_x_education'
    ]
    
    agg_dict = {col: ['mean', 'std', 'min', 'max', 'median', q25, q75, 'sum']
                for col in numeric_cols}
    bag_num = gb[numeric_cols].agg(agg_dict)
    bag_num.columns = [f"{c}_{s}" for c, s in bag_num.columns]
    
    # Range for key numerics
    for col in numeric_cols:
        bag_num[f"{col}_range"] = bag_num[f"{col}_max"] - bag_num[f"{col}_min"]
    
    # Binary indicators: proportion, count, any
    binary_cols = [
        'is_married', 'is_higher_edu', 'is_full_time', 'is_senior',
        'has_capital_gain', 'has_capital_loss',
        'is_exec_managerial', 'is_prof_specialty', 'is_self_employed', 'is_govt',
        'is_child', 'is_husband', 'is_wife'
    ]
    bag_bin = gb[binary_cols].agg(['mean', 'sum', 'max'])
    bag_bin.columns = [f"{c}_{s}" for c, s in bag_bin.columns]
    
    # Composition proportions
    props = pd.DataFrame(index=gb.groups.keys())
    props.index.name = 'bag_id'
    
    props['pct_male'] = gb.apply(lambda x: (x['sex'] == 'Male').mean())
    props['pct_female'] = gb.apply(lambda x: (x['sex'] == 'Female').mean())
    
    for race in ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']:
        safe = race.lower().replace('-', '_')
        props[f'pct_race_{safe}'] = gb.apply(lambda x, r=race: (x['race'] == r).mean())
    
    props['pct_us_born'] = gb.apply(lambda x: (x['native_country'] == 'United-States').mean())
    props['pct_private'] = gb.apply(lambda x: (x['workclass'] == 'Private').mean())
    props['pct_self_employed'] = gb.apply(lambda x: x['workclass'].isin(['Self-emp-inc', 'Self-emp-not-inc']).mean())
    props['pct_govt'] = gb.apply(lambda x: x['workclass'].isin(['Federal-gov', 'Local-gov', 'State-gov']).mean())
    
    rels = ['Husband', 'Wife', 'Own-child', 'Not-in-family', 'Unmarried', 'Other-relative']
    for rel in rels:
        safe = rel.lower().replace('-', '_')
        props[f'pct_rel_{safe}'] = gb.apply(lambda x, r=rel: (x['relationship'] == r).mean())
    
    marrs = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed',
             'Married-spouse-absent', 'Married-AF-spouse']
    for ms in marrs:
        safe = ms.lower().replace('-', '_').replace('.', '_')
        props[f'pct_marital_{safe}'] = gb.apply(lambda x, m=ms: (x['marital_status'] == m).mean())
    
    for tier in ['Primary', 'Secondary', 'Higher']:
        props[f'pct_edu_tier_{tier.lower()}'] = gb.apply(lambda x, t=tier: (x['education_tier'] == t).mean())
    
    middle_occ = ['Adm-clerical', 'Sales', 'Craft-repair', 'Tech-support',
                  'Protective-serv', 'Machine-op-inspct', 'Transport-moving']
    service_occ = ['Handlers-cleaners', 'Other-service', 'Priv-house-serv', 'Farming-fishing']
    props['pct_middle_occ'] = gb.apply(lambda x: x['occupation'].isin(middle_occ).mean())
    props['pct_service_occ'] = gb.apply(lambda x: x['occupation'].isin(service_occ).mean())
    
    # Diversity
    diversity = pd.DataFrame(index=gb.groups.keys())
    diversity.index.name = 'bag_id'
    diversity['nunique_occupation'] = gb['occupation'].nunique()
    diversity['nunique_workclass'] = gb['workclass'].nunique()
    diversity['nunique_education'] = gb['education'].nunique()
    diversity['nunique_relationship'] = gb['relationship'].nunique()
    diversity['nunique_marital_status'] = gb['marital_status'].nunique()
    diversity['nunique_race'] = gb['race'].nunique()
    
    # Household structure
    structure = pd.DataFrame(index=gb.groups.keys())
    structure.index.name = 'bag_id'
    structure['num_children'] = gb['is_child'].sum()
    structure['num_husband_wife'] = gb['is_husband'].sum() + gb['is_wife'].sum()
    structure['num_higher_edu'] = gb['is_higher_edu'].sum()
    structure['num_full_time'] = gb['is_full_time'].sum()
    structure['num_senior'] = gb['is_senior'].sum()
    structure['num_with_capital_gain'] = gb['has_capital_gain'].sum()
    structure['num_with_capital_loss'] = gb['has_capital_loss'].sum()
    
    has_husband = gb['is_husband'].max()
    has_wife = gb['is_wife'].max()
    structure['has_married_couple'] = ((has_husband > 0) & (has_wife > 0)).astype(int)
    structure['has_children'] = gb['is_child'].max()
    structure['single_parent'] = ((structure['has_children'] == 1) & (structure['has_married_couple'] == 0)).astype(int)
    structure['only_adults'] = 1 - structure['has_children']
    
    edu_range = gb['education_num'].max() - gb['education_num'].min()
    structure['education_range'] = edu_range
    structure['has_mixed_education'] = (edu_range >= 5).astype(int)
    structure['has_mixed_workclass'] = (gb['workclass'].nunique() > 1).astype(int)
    
    # Middle-class boundary features
    boundary = pd.DataFrame(index=gb.groups.keys())
    boundary.index.name = 'bag_id'
    for col in ['education_num', 'hours_per_week', 'age', 'annual_hours_est']:
        mean_vals = gb[col].mean()
        median_vals = gb[col].median()
        std_vals = gb[col].std().fillna(0)
        boundary[f'cv_{col}'] = std_vals / (mean_vals.abs() + 1e-6)
        boundary[f'mean_minus_median_{col}'] = mean_vals - median_vals
    
    boundary['num_part_time'] = gb.apply(lambda x: (x['hours_per_week'] < 35).sum())
    boundary['num_overtime'] = gb.apply(lambda x: (x['hours_per_week'] > 50).sum())
    boundary['all_zero_capital'] = (gb['capital_gain'].max() == 0).astype(int)
    
    bag = bag_num.join(bag_bin).join(props).join(diversity).join(structure).join(boundary)
    
    if is_train:
        labels = gb['label'].first().to_frame()
        bag = bag.join(labels)
    return bag.reset_index()

print("\nAggregating bags...")
train_bags = create_bag_features(train, is_train=True)
test_bags = create_bag_features(test, is_train=False)

# Align columns
train_cols = set(train_bags.columns)
test_cols = set(test_bags.columns)
for col in train_cols - test_cols - {'label'}:
    test_bags[col] = 0.0
extra = test_cols - train_cols
if extra:
    test_bags = test_bags.drop(columns=list(extra))
test_bags = test_bags[[c for c in train_bags.columns if c != 'label']]

print(f"Train bags: {train_bags.shape}  |  Test bags: {test_bags.shape}")

# =============================================================================
# 5. PREPARE FEATURES
# =============================================================================
feature_cols = [c for c in train_bags.columns if c not in ['bag_id', 'label']]
X = train_bags[feature_cols].values.astype(np.float32)
y = train_bags['label'].values
X_test = test_bags[feature_cols].values.astype(np.float32)

class_weights = {0: 1.1, 1: 1.4, 2: 1.0}
sample_wts = np.array([class_weights[int(v)] for v in y])

# =============================================================================
# 6. 5-FOLD OOF + THRESHOLD TUNING
# =============================================================================
print("\n5-Fold OOF training + threshold search...")
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_xgb = np.zeros((len(X), 3))
oof_cbt = np.zeros((len(X), 3))

for fold, (tr_idx, val_idx) in enumerate(SKF.split(X, y)):
    print(f"  Fold {fold+1}...", end=" ")
    Xtr, Xval = X[tr_idx], X[val_idx]
    ytr, yval = y[tr_idx], y[val_idx]
    wtr = sample_wts[tr_idx]
    
    m_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, max_depth=5,
        learning_rate=0.05, n_estimators=2000, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3, reg_alpha=0.5,
        min_child_weight=3, random_state=SEED, n_jobs=4, verbosity=0,
        eval_metric='mlogloss', early_stopping_rounds=100
    )
    m_xgb.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict_proba(Xval)
    
    m_cbt = CatBoostClassifier(
        loss_function='MultiClass', eval_metric='TotalF1',
        class_weights=class_weights, depth=6, learning_rate=0.05,
        iterations=2000, early_stopping_rounds=100, l2_leaf_reg=6,
        random_seed=SEED, verbose=False
    )
    m_cbt.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
    oof_cbt[val_idx] = m_cbt.predict_proba(Xval)
    print("Done")

f1_xgb = f1_score(y, oof_xgb.argmax(1), average='macro')
f1_cbt = f1_score(y, oof_cbt.argmax(1), average='macro')
print(f"\nOOF  XGB={f1_xgb:.4f}  CBT={f1_cbt:.4f}")

# Ensemble
best_wx = 0.95
base_proba = best_wx * oof_xgb + (1 - best_wx) * oof_cbt
base_f1 = f1_score(y, base_proba.argmax(1), average='macro')
print(f"Base ensemble F1: {base_f1:.4f}")

# Coarse + Fine multiplicative threshold search
print("\nSearching thresholds...")
best_f1 = base_f1
best_s0, best_s1, best_s2 = 1.0, 1.0, 1.0

# Coarse
for s0 in np.arange(1.0, 3.01, 0.10):
    for s1 in np.arange(0.5, 2.51, 0.10):
        for s2 in np.arange(0.2, 1.51, 0.05):
            scaled = base_proba * np.array([s0, s1, s2])
            f1 = f1_score(y, scaled.argmax(1), average='macro')
            if f1 > best_f1:
                best_f1, best_s0, best_s1, best_s2 = f1, s0, s1, s2

# Fine around best
for s0 in np.arange(max(1.0, best_s0 - 0.30), best_s0 + 0.31, 0.02):
    for s1 in np.arange(max(0.3, best_s1 - 0.30), best_s1 + 0.31, 0.02):
        for s2 in np.arange(max(0.1, best_s2 - 0.30), best_s2 + 0.31, 0.02):
            scaled = base_proba * np.array([s0, s1, s2])
            f1 = f1_score(y, scaled.argmax(1), average='macro')
            if f1 > best_f1:
                best_f1, best_s0, best_s1, best_s2 = f1, s0, s1, s2

print(f"Best scales: lower={best_s0:.3f} middle={best_s1:.3f} upper={best_s2:.3f}")
print(f"Tuned Macro F1: {best_f1:.4f}  (gain: +{best_f1 - base_f1:.4f})")

scaled = base_proba * np.array([best_s0, best_s1, best_s2])
print("\n" + classification_report(y, scaled.argmax(1), target_names=['lower','middle','upper'], digits=4))

# =============================================================================
# 7. FINAL MODELS ON 100% DATA
# =============================================================================
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

# =============================================================================
# 8. PREDICT TEST + SAVE
# =============================================================================
test_proba = 0.95 * final_xgb.predict_proba(X_test) + 0.05 * final_cbt.predict_proba(X_test)
test_proba[:, 0] *= best_s0
test_proba[:, 1] *= best_s1
test_proba[:, 2] *= best_s2
test_preds = test_proba.argmax(1)

submission = pd.DataFrame({
    'bag_id': test_bags['bag_id'].values,
    'label': test_preds
})
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
print(f"\nsubmission.csv saved: {len(submission)} rows")
print(f"Label dist: {dict(submission['label'].value_counts().sort_index())}")

# Save models
final_xgb.save_model(os.path.join(OUTPUT_DIR, 'final_xgb.json'))
final_cbt.save_model(os.path.join(OUTPUT_DIR, 'final_catboost.cbm'))
print("Models saved: final_xgb.json, final_catboost.cbm")

print("\n>>> NOTEBOOK COMPLETE <<<")
