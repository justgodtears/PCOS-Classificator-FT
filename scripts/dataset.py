from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import AutoImageProcessor

SPLITS_CSV = Path("../dataset/ops/splits.csv")
MODEL_NAME = "facebook/dinov2-small"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# mean/std using the model's processor so that the normalization matches the pretraining
_proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
NORMALIZE = T.Normalize(mean=_proc.image_mean, std=_proc.image_std)

# train -  random crop and light augmentation
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    NORMALIZE,
])

# val/test only resize, without randomness
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    NORMALIZE,
])


class PcosDataset(Dataset):
    # returns (image tensor, label_id) for row in splits.csv
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def _loader(df, transform, shuffle):
    return DataLoader(
        PcosDataset(df, transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )


def get_fold_loaders(fold):
    # train - all folds except 'fold', val = this fold
    df = pd.read_csv(SPLITS_CSV)
    tr = df[(df["split"] == "train") & (df["fold"] != fold)]
    va = df[(df["split"] == "train") & (df["fold"] == fold)]
    return _loader(tr, train_tf, True), _loader(va, eval_tf, False)


def get_full_train_loader():
    # all train ( for final model)
    df = pd.read_csv(SPLITS_CSV)
    return _loader(df[df["split"] == "train"], train_tf, True)


def get_test_loader():
    # for testing
    df = pd.read_csv(SPLITS_CSV)
    return _loader(df[df["split"] == "test"], eval_tf, False)


if __name__ == "__main__":
    # sanity check
    train_loader, val_loader = get_fold_loaders(0)
    print("train batches:", len(train_loader), "val batches:", len(val_loader))

    x, y = next(iter(train_loader))
    print("x:", x.shape, x.dtype, "| y:", y.shape, y.dtype)
    print("x min/max/mean/std:", x.min().item(), x.max().item(), x.mean().item(), x.std().item())
    print("labels in batch:", torch.bincount(y, minlength=2).tolist())

    xv, yv = next(iter(val_loader))
    print("val x:", xv.shape, "| labels:", torch.bincount(yv, minlength=2).tolist())

