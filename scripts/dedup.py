import csv
from pathlib import Path

from src.pcos_ft.deduplication import deduplicate_images, create_tagged_list

DATASET_PATH = Path(r"C:\Users\justg\Desktop\Development\PCOS FT\dataset\raw")
ALL_IMAGES = [str(f) for f in DATASET_PATH.rglob("*") if f.is_file()]
SAVING_PATH = Path("../dataset/ops/dedup.csv")


dedup_images = deduplicate_images(ALL_IMAGES)
tagged_list = create_tagged_list(dedup_images)

with open(SAVING_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    for data in tagged_list:
        writer.writerow(data)


