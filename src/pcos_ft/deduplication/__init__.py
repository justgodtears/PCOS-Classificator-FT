import imagehash
from PIL import Image
from pathlib import Path, PurePath


def count_phash_add_path(img_path):
    image = Image.open(img_path)
    img_hash = imagehash.phash(image, 24)
    return img_hash, img_path


def deduplicate_images(img_paths):
    unique_data = []
    seen_hashes = set()

    for  path in img_paths:
        phash, img_path = count_phash_add_path(path)
        hash_str = str(phash)

        if hash_str in seen_hashes:
            continue

        img = Image.open(path)

        hash_90 = str(imagehash.phash(img.rotate(90, expand=True), 24))
        hash_180 = str(imagehash.phash(img.rotate(180, expand=True), 24))
        hash_270 = str(imagehash.phash(img.rotate(270, expand=True), 24))

        img.close()

        rotated_hashes = {hash_90, hash_180, hash_270}

        if seen_hashes.intersection(rotated_hashes):
            continue

        unique_data.append((phash, path))

        seen_hashes.add(hash_str)
        seen_hashes.update(rotated_hashes)

    return unique_data


def create_tagged_list(dedup_paths):
    tagged_list = []

    for phash, path in dedup_paths:
        path_object = Path(path)

        if path_object.parent.name == "infected":
            tagged_list.append((path, "infected"))
        else:
            tagged_list.append((path, "noninfected"))

    return tagged_list