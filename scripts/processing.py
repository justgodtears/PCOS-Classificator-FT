import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

DATA = Path("../dataset/ops/dedup.csv")
OUT_DIR = DATA.parent
LABEL2ID = {"noninfected": 0, "infected": 1}

df = pd.read_csv(DATA, header=None, names=["path", "label"])

# Split
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# fold for every train row
train_df["fold"] = -1
fold_index = train_df.columns.get_loc("fold")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
    train_df.iloc[val_idx, fold_index] = i

# class distribution
print("TEST")
print(test_df["label"].value_counts())
print(test_df["label"].value_counts(normalize=True))

print("FOLD")
print(pd.crosstab(train_df["fold"], train_df["label"]))
print(pd.crosstab(train_df["fold"], train_df["label"], normalize="index"))

assert (train_df["fold"] >= 0).all()

# csv with split and fold
train_df["split"] = "train"
test_df["split"] = "test"
test_df["fold"] = -1
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df["label_id"] = full_df["label"].map(LABEL2ID)
assert full_df["label_id"].notna().all()
full_df["label_id"] = full_df["label_id"].astype(int)
full_df.to_csv(OUT_DIR / "splits.csv", index=False)

# class weights from train
train_ids = full_df.loc[full_df["split"] == "train", "label_id"].to_numpy()
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_ids)
with open(OUT_DIR / "class_weights.json", "w") as f:
    json.dump({"label2id": LABEL2ID, "weights": weights.tolist()}, f, indent=2)

# saving control
chk = pd.read_csv(OUT_DIR / "splits.csv")
print("SAVE")
print(len(chk))
print(chk.groupby("split")["label"].value_counts())
print("weights [noninfected, infected]:", weights)