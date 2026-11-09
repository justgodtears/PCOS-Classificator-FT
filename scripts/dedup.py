import csv
from pathlib import Path

from src.pcos_ft.deduplication import deduplicate_images, create_tagged_list

dataset_path = Path(r"C:\Users\justg\Desktop\Development\PCOS FT\dataset\raw")
all_images = [str(f) for f in dataset_path.rglob("*") if f.is_file()]
saving_path = Path("../dataset/ops/dedup.csv")


dedup_images = deduplicate_images(all_images)
tagged_list = create_tagged_list(dedup_images)

with open(saving_path, "w", newline="") as f:
    writer = csv.writer(f)
    for data in tagged_list:
        writer.writerow(data)

