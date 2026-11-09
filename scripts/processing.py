from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

DATA = Path("../dataset/ops/dedup.csv")

df = pd.read_csv(DATA, header=None, names=["path", "label"])

train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df["fold"] = -1
fold_index = train_df.columns.get_loc("fold")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i,(x,y) in enumerate(skf.split(train_df, train_df["label"])):
    train_df.iloc[y, fold_index] = i

print(train_df["fold"].value_counts())
print(train_df.groupby("fold")["label"].value_counts(normalize=True))