"""
Step 1: Preprocess Raw Rows
===========================
WHY THIS FILE:
  - The raw CSV contains constant columns (survey_year=1994 everywhere), 
    leakage features (interviewer_id), and redundant flags (capital_activity_flag).
  - We must clean these BEFORE aggregation so the model never sees them.
  - We also create per-person interaction features (age, log capital, hours*education, 
    binary indicators) here because they must be computed at row level before bag aggregation.
  - Finally, we map string labels ('lower','middle','upper') to integers (0,1,2) 
    so CatBoost/XGBoost can train.

WHY RUN IT:
  - This is the foundation. Every later step reads the clean CSVs produced here.
  - You should inspect train_rows_clean.csv to verify derived features look correct.

COMMAND TO RUN:
  python step1_preprocess_rows.py

OUTPUTS:
  data/train_rows_clean.csv
  data/test_rows_clean.csv
"""

import pandas as pd
import numpy as np
import os

# Make sure output directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("STEP 1: Loading raw data...")
print("=" * 60)

train = pd.read_csv('Coderush-26-ML-Train.csv')
test = pd.read_csv('Coderush-26-ML-test.csv')

print(f"Raw train shape: {train.shape}")
print(f"Raw test shape:  {test.shape}")

# ------------------------------------------------------------------
# 1. DROP LEAKAGE / CONSTANT / REDUNDANT COLUMNS
# ------------------------------------------------------------------
DROP_COLS = [
    'survey_year',        # constant 1994
    'currency_code',      # constant USD
    'poverty_line_usd',   # constant 15141
    'processing_flag',    # constant 1.0
    'interview_mode',     # constant 'in-person'
    'capital_activity_flag',  # exactly (net_capital_asset != 0), redundant
    'interviewer_id',     # 500 IDs, massive overfit risk on hidden test
    'person_idx',         # unordered bag, position must not be used
]

print(f"\nDropping: {DROP_COLS}")
train = train.drop(columns=DROP_COLS)
test = test.drop(columns=DROP_COLS)

# ------------------------------------------------------------------
# 2. LABEL ENCODING (0/1/2)
# ------------------------------------------------------------------
LABEL_MAP = {'lower': 0, 'middle': 1, 'upper': 2}
train['label'] = train['label'].map(LABEL_MAP)
print(f"\nLabel mapping: {LABEL_MAP}")
print(f"Label distribution:\n{train['label'].value_counts().sort_index()}")

# ------------------------------------------------------------------
# 3. DERIVE PER-PERSON NUMERIC FEATURES
# ------------------------------------------------------------------
print("\nDeriving numeric features...")

for df in [train, test]:
    # Age (survey year was 1994)
    df['age'] = 1994 - df['year_of_birth']
    
    # Log transform for heavily skewed capital columns
    df['log_capital_gain'] = np.log1p(df['capital_gain'])
    df['log_capital_loss'] = np.log1p(df['capital_loss'])
    
    # Interaction features
    df['hours_x_education'] = df['hours_per_week'] * df['education_num']
    df['age_x_education'] = df['age'] * df['education_num']

# Drop original year_of_birth (redundant with age)
train = train.drop(columns=['year_of_birth'])
test = test.drop(columns=['year_of_birth'])

# ------------------------------------------------------------------
# 4. DERIVE PER-PERSON BINARY INDICATORS
#    These will become household proportions after bag aggregation.
# ------------------------------------------------------------------
print("Deriving binary indicators...")

for df in [train, test]:
    # Demographic / family
    df['is_married'] = (df['marital_status'] == 'Married-civ-spouse').astype(int)
    df['is_child'] = (df['relationship'] == 'Own-child').astype(int)
    df['is_husband'] = (df['relationship'] == 'Husband').astype(int)
    df['is_wife'] = (df['relationship'] == 'Wife').astype(int)
    
    # Education & work
    df['is_higher_edu'] = (df['education_num'] >= 13).astype(int)
    df['is_full_time'] = (df['hours_per_week'] >= 40).astype(int)
    df['is_senior'] = (df['age'] >= 60).astype(int)
    
    # Economic
    df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'] > 0).astype(int)
    
    # Occupation / sector
    df['is_exec_managerial'] = (df['occupation'] == 'Exec-managerial').astype(int)
    df['is_prof_specialty'] = (df['occupation'] == 'Prof-specialty').astype(int)
    df['is_self_employed'] = df['workclass'].isin(['Self-emp-inc', 'Self-emp-not-inc']).astype(int)
    df['is_govt'] = df['workclass'].isin(['Federal-gov', 'Local-gov', 'State-gov']).astype(int)

# ------------------------------------------------------------------
# 5. SAVE CLEAN ROW-LEVEL DATA
# ------------------------------------------------------------------
train_path = 'data/train_rows_clean.csv'
test_path = 'data/test_rows_clean.csv'

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"Saved: {train_path} | shape: {train.shape}")
print(f"Saved: {test_path}  | shape: {test.shape}")
print("\nNew derived columns in both files:")
new_cols = [
    'age', 'log_capital_gain', 'log_capital_loss',
    'hours_x_education', 'age_x_education',
    'is_married', 'is_higher_edu', 'is_full_time', 'is_senior',
    'has_capital_gain', 'has_capital_loss',
    'is_exec_managerial', 'is_prof_specialty', 'is_self_employed',
    'is_govt', 'is_child', 'is_husband', 'is_wife'
]
print(new_cols)
print("\nNext: Run step2_bag_aggregation.py")
