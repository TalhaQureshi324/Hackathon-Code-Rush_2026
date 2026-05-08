"""
Step 2: Bag-Level Feature Engineering
=====================================
WHY THIS FILE:
  - The problem is Multi-Instance Learning: each bag (household) has 3-7 unordered people.
  - We must collapse each bag into a SINGLE feature vector using ONLY 
    permutation-invariant operations (mean, std, median, max, proportions, counts).
  - This step creates ~120-150 bag-level features that capture household economics.

WHY RUN IT:
  - Step 3 (modeling) reads the output CSVs from here.
  - You must verify that train and test bags have the exact same columns.
  - This is where 80% of your accuracy comes from.

COMMAND TO RUN:
  python step2_bag_aggregation.py

OUTPUTS:
  data/train_bags.csv
  data/test_bags.csv
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 2: Loading cleaned row-level data...")
print("=" * 70)

train_rows = pd.read_csv('data/train_rows_clean.csv')
test_rows = pd.read_csv('data/test_rows_clean.csv')

print(f"Train rows: {train_rows.shape}")
print(f"Test rows:  {test_rows.shape}")


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def create_bag_features(df, is_train=True):
    """
    Aggregate person-level rows into bag-level features.
    Every operation is invariant to row ordering within the bag.
    """
    gb = df.groupby('bag_id')

    # ==================================================================
    # 1. ROBUST NUMERIC STATS
    # ==================================================================
    numeric_cols = [
        'bag_size', 'education_num', 'survey_duration_mins',
        'capital_gain', 'capital_loss', 'hours_per_week', 'net_capital_asset',
        'is_adult_flag', 'annual_hours_est',
        'age', 'log_capital_gain', 'log_capital_loss',
        'hours_x_education', 'age_x_education'
    ]

    agg_dict = {col: ['mean', 'std', 'min', 'max', 'median', q25, q75, 'sum']
                for col in numeric_cols}

    bag_num = gb[numeric_cols].agg(agg_dict)
    bag_num.columns = [f"{c}_{s}" for c, s in bag_num.columns]

    # Range (max - min) for key numerics — critical for small bags
    for col in numeric_cols:
        bag_num[f"{col}_range"] = bag_num[f"{col}_max"] - bag_num[f"{col}_min"]

    # ==================================================================
    # 2. BINARY INDICATOR AGGREGATES (proportion + count + any)
    # ==================================================================
    binary_cols = [
        'is_married', 'is_higher_edu', 'is_full_time', 'is_senior',
        'has_capital_gain', 'has_capital_loss',
        'is_exec_managerial', 'is_prof_specialty', 'is_self_employed', 'is_govt',
        'is_child', 'is_husband', 'is_wife'
    ]

    bag_bin = gb[binary_cols].agg(['mean', 'sum', 'max'])
    bag_bin.columns = [f"{c}_{s}" for c, s in bag_bin.columns]

    # ==================================================================
    # 3. EXPLICIT COMPOSITION PROPORTIONS
    # ==================================================================
    props = pd.DataFrame(index=gb.groups.keys())
    props.index.name = 'bag_id'

    # --- Gender ---
    props['pct_male'] = gb.apply(lambda x: (x['sex'] == 'Male').mean())
    props['pct_female'] = gb.apply(lambda x: (x['sex'] == 'Female').mean())

    # --- Race ---
    for race in ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']:
        safe = race.lower().replace('-', '_')
        props[f'pct_race_{safe}'] = gb.apply(lambda x, r=race: (x['race'] == r).mean())

    # --- Native country ---
    props['pct_us_born'] = gb.apply(lambda x: (x['native_country'] == 'United-States').mean())

    # --- Workclass ---
    props['pct_private'] = gb.apply(lambda x: (x['workclass'] == 'Private').mean())
    props['pct_self_employed'] = gb.apply(lambda x: x['workclass'].isin(
        ['Self-emp-inc', 'Self-emp-not-inc']).mean())
    props['pct_govt'] = gb.apply(lambda x: x['workclass'].isin(
        ['Federal-gov', 'Local-gov', 'State-gov']).mean())

    # --- Relationship ---
    rels = ['Husband', 'Wife', 'Own-child', 'Not-in-family', 'Unmarried', 'Other-relative']
    for rel in rels:
        safe = rel.lower().replace('-', '_')
        props[f'pct_rel_{safe}'] = gb.apply(lambda x, r=rel: (x['relationship'] == r).mean())

    # --- Marital status ---
    marrs = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed',
             'Married-spouse-absent', 'Married-AF-spouse']
    for ms in marrs:
        safe = ms.lower().replace('-', '_').replace('.', '_')
        props[f'pct_marital_{safe}'] = gb.apply(lambda x, m=ms: (x['marital_status'] == m).mean())

    # --- Education tier ---
    for tier in ['Primary', 'Secondary', 'Higher']:
        props[f'pct_edu_tier_{tier.lower()}'] = gb.apply(
            lambda x, t=tier: (x['education_tier'] == t).mean())

    # --- Occupation classes ---
    middle_occ = ['Adm-clerical', 'Sales', 'Craft-repair', 'Tech-support',
                  'Protective-serv', 'Machine-op-inspct', 'Transport-moving']
    service_occ = ['Handlers-cleaners', 'Other-service', 'Priv-house-serv', 'Farming-fishing']

    props['pct_middle_occ'] = gb.apply(lambda x: x['occupation'].isin(middle_occ).mean())
    props['pct_service_occ'] = gb.apply(lambda x: x['occupation'].isin(service_occ).mean())

    # ==================================================================
    # 4. DIVERSITY / NUNIQUE FEATURES
    # ==================================================================
    diversity = pd.DataFrame(index=gb.groups.keys())
    diversity.index.name = 'bag_id'
    diversity['nunique_occupation'] = gb['occupation'].nunique()
    diversity['nunique_workclass'] = gb['workclass'].nunique()
    diversity['nunique_education'] = gb['education'].nunique()
    diversity['nunique_relationship'] = gb['relationship'].nunique()
    diversity['nunique_marital_status'] = gb['marital_status'].nunique()
    diversity['nunique_race'] = gb['race'].nunique()

    # ==================================================================
    # 5. HOUSEHOLD STRUCTURE FEATURES
    # ==================================================================
    structure = pd.DataFrame(index=gb.groups.keys())
    structure.index.name = 'bag_id'

    # Raw counts
    structure['num_children'] = gb['is_child'].sum()
    structure['num_husband_wife'] = gb['is_husband'].sum() + gb['is_wife'].sum()
    structure['num_higher_edu'] = gb['is_higher_edu'].sum()
    structure['num_full_time'] = gb['is_full_time'].sum()
    structure['num_senior'] = gb['is_senior'].sum()
    structure['num_with_capital_gain'] = gb['has_capital_gain'].sum()
    structure['num_with_capital_loss'] = gb['has_capital_loss'].sum()

    # Household type flags
    has_husband = gb['is_husband'].max()
    has_wife = gb['is_wife'].max()
    structure['has_married_couple'] = ((has_husband > 0) & (has_wife > 0)).astype(int)
    structure['has_children'] = gb['is_child'].max()
    structure['single_parent'] = ((structure['has_children'] == 1) & (
        structure['has_married_couple'] == 0)).astype(int)
    structure['only_adults'] = (1 - structure['has_children'])

    # Education spread (mixed generations = middle-class signal)
    edu_min = gb['education_num'].min()
    edu_max = gb['education_num'].max()
    structure['education_range'] = edu_max - edu_min
    structure['has_mixed_education'] = (structure['education_range'] >= 5).astype(int)

    # Workclass spread
    structure['has_mixed_workclass'] = (gb['workclass'].nunique() > 1).astype(int)

    # ==================================================================
    # 6. MIDDLE-CLASS BOUNDARY FEATURES
    # ------------------------------------------------------------------
    # The middle class is hardest to separate. These features capture
    # "moderateness", variability, and household mixing.
    # ==================================================================
    boundary = pd.DataFrame(index=gb.groups.keys())
    boundary.index.name = 'bag_id'

    for col in ['education_num', 'hours_per_week', 'age', 'annual_hours_est']:
        mean_vals = gb[col].mean()
        median_vals = gb[col].median()
        std_vals = gb[col].std().fillna(0)

        # Coefficient of variation: high CV = mixed household (middle class often mixed)
        boundary[f'cv_{col}'] = std_vals / (mean_vals.abs() + 1e-6)

        # Mean - median skew: upper class often skewed by one high earner
        boundary[f'mean_minus_median_{col}'] = mean_vals - median_vals

    # Hours distribution signals
    boundary['num_part_time'] = gb.apply(lambda x: (x['hours_per_week'] < 35).sum())
    boundary['num_overtime'] = gb.apply(lambda x: (x['hours_per_week'] > 50).sum())

    # Capital polarization: all zero = lower/middle steady; any nonzero = potential upper signal
    boundary['all_zero_capital'] = (gb['capital_gain'].max() == 0).astype(int)

    # ==================================================================
    # 7. ASSEMBLE FINAL BAG DATAFRAME
    # ==================================================================
    bag = bag_num.join(bag_bin).join(props).join(diversity).join(structure).join(boundary)

    # Attach label (only for train)
    if is_train:
        labels = gb['label'].first().to_frame()
        bag = bag.join(labels)

    return bag.reset_index()


print("\nBuilding train bag features...")
train_bags = create_bag_features(train_rows, is_train=True)

print("Building test bag features...")
test_bags = create_bag_features(test_rows, is_train=False)

# ==================================================================
# 8. ALIGN COLUMNS: test must have exactly the same features as train
# ==================================================================
train_cols = set(train_bags.columns)
test_cols = set(test_bags.columns)

# Add missing columns to test (fill with 0)
missing_in_test = train_cols - test_cols - {'label'}
for col in missing_in_test:
    test_bags[col] = 0.0

# Drop any extra columns in test that aren't in train
extra_in_test = test_cols - train_cols
if extra_in_test:
    test_bags = test_bags.drop(columns=list(extra_in_test))

# Reorder test columns to match train (excluding label)
train_feature_cols = [c for c in train_bags.columns if c != 'label']
test_bags = test_bags[train_feature_cols]

# Verify
assert list(train_bags.drop(columns='label').columns) == list(test_bags.columns), \
    "Column mismatch between train and test!"

# ==================================================================
# 9. SAVE
# ==================================================================
train_bags.to_csv('data/train_bags.csv', index=False)
test_bags.to_csv('data/test_bags.csv', index=False)

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)
print(f"Train bags: {train_bags.shape}  ({train_bags.shape[1]-2} features + bag_id + label)")
print(f"Test bags:  {test_bags.shape}  ({test_bags.shape[1]-1} features + bag_id)")
print(f"\nLabel distribution:")
print(train_bags['label'].value_counts().sort_index())
print("\nNext: Run step3_create_folds.py")
