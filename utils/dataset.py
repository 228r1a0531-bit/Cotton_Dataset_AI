"""Discover, verify, and split local cotton leaf images."""

import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps

from config import ROOT

CLASSES = ("bacterial_blight", "curl_virus", "fusarium_wilt", "healthy")
SPLITS = ("train", "validation", "test")
RATIO = (70, 15, 15)
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
ALIASES = {
    "bacterialblight": "bacterial_blight",
    "bacteriablight": "bacterial_blight",
    "curlvirus": "curl_virus",
    "leafcurl": "curl_virus",
    "leafcurlvirus": "curl_virus",
    "cottonleafcurlvirus": "curl_virus",
    "fusariumwilt": "fusarium_wilt",
    "fussariumwilt": "fusarium_wilt",
    "healthy": "healthy",
    "healthyleaf": "healthy",
    "healthyleaves": "healthy",
}


@dataclass(frozen=True)
class Sample:
    path: str
    label: str
    digest: str


@dataclass
class Scan:
    root: Path
    images: list[Sample]
    unreadable: dict[str, str]
    matched_classes: set[str]

    @property
    def duplicates(self) -> list[list[Sample]]:
        groups = defaultdict(list)
        for image in self.images:
            groups[image.digest].append(image)
        return [group for group in groups.values() if len(group) > 1]


def class_name(folder: Path) -> str | None:
    return ALIASES.get(re.sub(r"[^a-z]", "", folder.name.casefold()))


def discover(path: str = "") -> Path | None:
    """Resolve an explicit root or search common repository dataset locations."""
    configured = path.strip() or os.environ.get("COTTONLEAF_DATASET_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser()
        return (ROOT / root).resolve()

    partial = None
    for name in ("datasets", "dataset", "data"):
        location = ROOT / name
        if location.is_dir():
            for candidate in [location, *sorted(location.iterdir())]:
                if candidate.is_dir():
                    labels = {class_name(child) for child in candidate.iterdir() if child.is_dir()}
                    if set(CLASSES) <= labels:
                        return candidate
                    if partial is None and labels.intersection(CLASSES):
                        partial = candidate
    return partial


def scan(root: Path) -> Scan:
    root = root.resolve()
    result = Scan(root, [], {}, set())
    for folder in sorted(root.iterdir()):
        label = class_name(folder)
        if not folder.is_dir() or label is None:
            continue
        result.matched_classes.add(label)
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    pixels = ImageOps.exif_transpose(image).convert("RGB")
                    digest = hashlib.sha256(f"{pixels.width}x{pixels.height}:".encode())
                    digest.update(pixels.tobytes())
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                result.unreadable[relative] = str(error)
            else:
                result.images.append(Sample(relative, label, digest.hexdigest()))
    return result


def allocation(total: int) -> list[int]:
    """Allocate integer counts close to the target, with a sample in every split."""
    counts = [total * share // 100 for share in RATIO]
    fractions = [total * share % 100 for share in RATIO]
    for index in sorted(range(3), key=lambda i: -fractions[i])[:total - sum(counts)]:
        counts[index] += 1
    for index, count in enumerate(counts):
        if count == 0:
            donor = max(range(3), key=counts.__getitem__)
            counts[donor] -= 1
            counts[index] = 1
    return counts


def stratify(dataset: Scan) -> dict[str, list[Sample]]:
    """Use one representative per exact pixel digest and exclude conflicting labels."""
    excluded = set()
    for group in dataset.duplicates:
        copies = group[1:] if len({image.label for image in group}) == 1 else group
        excluded.update(copy.path for copy in copies)

    rng = random.Random(SEED)
    result = {name: [] for name in SPLITS}
    for label in CLASSES:
        images = sorted(
            (image for image in dataset.images if image.label == label and image.path not in excluded),
            key=lambda image: image.path,
        )
        if len(images) < 3:
            raise ValueError(f"{label} needs at least three distinct, consistently labeled images.")
        rng.shuffle(images)
        offset = 0
        for name, count in zip(SPLITS, allocation(len(images))):
            result[name].extend(images[offset:offset + count])
            offset += count
    return result


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            write(temp)
            temp.close()
            temporary.replace(path)
        finally:
            temp.close()
            temporary.unlink(missing_ok=True)


def save_splits(directory: Path, dataset: Scan, splits: dict[str, list[Sample]]) -> None:
    metadata = {
        "root": str(dataset.root),
        "seed": SEED,
        "ratios": dict(zip(SPLITS, RATIO)),
        "splits": {name: [asdict(image) for image in images] for name, images in splits.items()},
    }
    atomic_write(directory / "splits.json", lambda handle: json.dump(metadata, handle, indent=2))

    def write_summary(handle):
        writer = csv.writer(handle)
        writer.writerow(("class", "split", "count"))
        for name, images in splits.items():
            writer.writerows((label, name, sum(image.label == label for image in images)) for label in CLASSES)

    atomic_write(directory / "dataset.csv", write_summary)
