"""
Generate Fixed Test Holdout split (wide-format, single assignment):

- 80% patients -> pool (được gán fold 1..10 theo GroupKFold, mỗi fold có patient dùng làm val)
- 20% patients -> test_fixed (fold=0, split=test_fixed, held-out)

Output: wide-format CSV, mỗi ảnh 1 dòng, ~1805 rows (KHÔNG duplicate).
- Pool rows:  fold ∈ {1..10}, split = "train" (placeholder)
- test_fixed: fold = 0, split = "test_fixed"

Vì split="train" là placeholder cho pool (runtime trong train.py sẽ dynamic resolve
val/test theo current_fold), CSV wide-format không thể encode sẵn val/test xoay vòng.
Train.py sẽ tự xử lý logic khi load dataloader.

Kiểu split khi train fold N (trong train.py):
    train_loader  = pool rows where fold ∈ {1..10} AND fold != N
    val_loader    = pool rows where fold == N
    test_in_pool  = pool rows where fold == test_fold (xoay vòng)
    test_fixed    = rows where fold == 0 (luôn cùng 1 tập)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SRC = "datasets/chd_jaundice_pure2.csv"
OUT = "datasets/split_holdout.csv"
SEED = 42
TEST_FIXED_RATIO = 0.20
N_FOLDS = 10

df = pd.read_csv(SRC)

# --- Bước 1: tách 20% test_fixed (GroupShuffleSplit) ---
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FIXED_RATIO, random_state=SEED)
pool_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
pool_df = df.iloc[pool_idx].copy()
test_fixed_df = df.iloc[test_idx].copy()

# --- Bước 2: 10-fold GroupKFold trên 80% pool ---
gkf = GroupKFold(n_splits=N_FOLDS)
fold_assignment = np.zeros(len(pool_df), dtype=int)
for fold_idx, (_, val_idx) in enumerate(gkf.split(pool_df, groups=pool_df["patient_id"]), 1):
    fold_assignment[val_idx] = fold_idx

pool_df["fold"] = fold_assignment
# Wide-format placeholder: runtime resolve train/val dựa trên current fold
pool_df["split"] = "train"

# test_fixed rows
test_fixed_df["fold"] = 0
test_fixed_df["split"] = "test_fixed"

# --- Ghép & xuất ---
out_df = pd.concat([pool_df, test_fixed_df], ignore_index=True)
out_df.to_csv(OUT, index=False)

# --- Log summary ---
print("=== Pool (80%, fold assignment) ===")
for fold in range(1, N_FOLDS + 1):
    n_pat = pool_df[pool_df["fold"] == fold]["patient_id"].nunique()
    n_img = (pool_df["fold"] == fold).sum()
    print(f"  Fold {fold:2d}: {n_pat} patients, {n_img} images")

print("\n=== test_fixed (20%) ===")
print(f"  {test_fixed_df['patient_id'].nunique()} patients, {len(test_fixed_df)} images")
print(f"  Label mean: {test_fixed_df['blood(mg/dL)'].mean():.2f} mg/dL")
print(f"  Treatment rate: {test_fixed_df['Treatment'].mean():.2f}")

print("\n=== Total ===")
print(f"  {out_df['patient_id'].nunique()} patients, {len(out_df)} images")

# --- Verify ---
all_pids = out_df["patient_id"].unique()
assert len(all_pids) == len(set(all_pids)), "Duplicate patient_id!"
overlap = set(pool_df["patient_id"]) & set(test_fixed_df["patient_id"])
assert len(overlap) == 0, f"Patient overlap: {overlap}"
print("\nOK: 0 patient overlap, all patients unique")
