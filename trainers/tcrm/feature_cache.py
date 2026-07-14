import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _extract_image_label(item):
    if isinstance(item, dict):
        label = int(item["label"])
        image = item.get("data", None)
        if image is None and "impath" in item:
            image = Image.open(item["impath"]).convert("RGB")
    elif hasattr(item, "data") and hasattr(item, "label"):
        image = item.data
        label = int(item.label)
    elif hasattr(item, "impath") and hasattr(item, "label"):
        image = Image.open(item.impath).convert("RGB")
        label = int(item.label)
    else:
        image, label = item[0], int(item[1])
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
    return image, label


def cache_metadata(dataset_name, split, backbone, sample_count, feature_dim=None, transform_name="deterministic_clip_eval", cache_version=2, clip_precision="fp32"):
    return {
        "dataset": str(dataset_name),
        "split": str(split),
        "backbone": str(backbone),
        "sample_count": int(sample_count),
        "feature_dim": None if feature_dim is None else int(feature_dim),
        "feature_cache_transform": str(transform_name),
        "clip_precision": str(clip_precision),
        "cache_version": int(cache_version),
    }


def _metadata_matches(old, expected):
    for key, value in expected.items():
        if key == "feature_dim" and value is None:
            continue
        if old.get(key) != value:
            return False
    return True


@torch.no_grad()
def build_or_load_feature_cache(
    data_source,
    transform,
    clip_model,
    cache_path,
    dataset_name,
    split,
    backbone,
    batch_size=128,
    device="cuda",
    dtype="float16",
    clip_precision="fp32",
    force_rebuild=False,
    log_fn=print,
):
    cache_path = Path(cache_path)
    expected = cache_metadata(dataset_name, split, backbone, len(data_source), clip_precision=clip_precision)
    if cache_path.exists() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and _metadata_matches(payload.get("metadata", {}), expected):
            if log_fn:
                log_fn(f"TCRM feature cache hit: {cache_path}")
            return payload["features"].float(), payload["labels"].long(), payload.get("metadata", {})
        if log_fn:
            log_fn(f"TCRM feature cache metadata mismatch, rebuilding: {cache_path}")
    if log_fn:
        log_fn(f"TCRM feature cache build: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    clip_model.eval().to(device)
    try:
        model_dtype = next(clip_model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    features = []
    labels = []
    batch_images = []
    for index in range(len(data_source)):
        image, label = _extract_image_label(data_source[index])
        batch_images.append(transform(image))
        labels.append(int(label))
        if len(batch_images) >= int(batch_size) or index == len(data_source) - 1:
            images = torch.stack(batch_images).to(device=device, dtype=model_dtype)
            feats = clip_model.encode_image(images)
            feats = F.normalize(feats.float(), dim=-1).detach().cpu()
            features.append(feats)
            batch_images = []
    features = torch.cat(features, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.long)
    save_features = features.half() if str(dtype).lower() == "float16" else features.float()
    metadata = cache_metadata(dataset_name, split, backbone, len(data_source), feature_dim=features.shape[1], clip_precision=clip_precision)
    torch.save({"metadata": metadata, "features": save_features, "labels": labels}, cache_path)
    return features.float(), labels, metadata


class FeatureSubset(Dataset):
    def __init__(self, features, labels, indices):
        self.features = features.float()
        self.labels = labels.long()
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[int(item)]
        return self.features[idx].float(), self.labels[idx].long(), idx
