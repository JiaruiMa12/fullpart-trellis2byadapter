from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type
import csv
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..configs.base_config import InstantiateConfig


@dataclass
class GlobalSLatPairDataConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: GlobalSLatPairData)
    dataset_csv_path: str = "dataset/partversexl/home_sample_1000.csv"
    old_global_slat_dir: str = "dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
    trellis2_global_slat_dir: str = "dataset/partversexl/trellis2_global_shape_latents"
    slat_norm: bool = True
    trellis2_slat_norm: bool = True
    trellis2_slat_stats_json: str = "dataset/partversexl/trellis2_shape_latents_stats.json"
    batch_size: int = 1
    num_processes: int = 4
    shuffle: bool = True
    num_samples: Optional[int] = None


class GlobalSLatPairDataset(Dataset):
    def __init__(self, config: GlobalSLatPairDataConfig):
        self.config = config
        self.old_global_slat_dir = Path(config.old_global_slat_dir)
        self.trellis2_global_slat_dir = Path(config.trellis2_global_slat_dir)
        self.ids = self._read_ids(Path(config.dataset_csv_path))
        if config.num_samples is not None:
            self.ids = self.ids[:config.num_samples]
        self.old_slat_normalization = {
            "mean": [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133, -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826],
            "std": [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695, 2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338],
        }
        self.trellis2_slat_normalization = None
        stats_path = Path(config.trellis2_slat_stats_json)
        if stats_path.exists():
            with stats_path.open("r") as f:
                self.trellis2_slat_normalization = json.load(f)

    def _read_ids(self, csv_path: Path):
        ids = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                asset_id = row.get("asset_id")
                if asset_id and row.get("skip", "False") == "False":
                    old_path = self.old_global_slat_dir / f"{asset_id}.npz"
                    t2_path = self.trellis2_global_slat_dir / f"{asset_id}.npz"
                    if old_path.exists() and t2_path.exists():
                        ids.append(asset_id)
        return ids

    def __len__(self):
        return len(self.ids)

    def _load_slat(self, path: Path, norm, do_norm: bool):
        data = np.load(path)
        feats = torch.from_numpy(data["feats"]).float()
        coords = torch.from_numpy(data["coords"]).int()
        if do_norm and norm is not None:
            mean = torch.tensor(norm["mean"])[None]
            std = torch.tensor(norm["std"])[None]
            feats = (feats - mean) / std
        return feats, coords

    def __getitem__(self, index):
        asset_id = self.ids[index]
        old_feats, old_coords = self._load_slat(
            self.old_global_slat_dir / f"{asset_id}.npz",
            self.old_slat_normalization,
            self.config.slat_norm,
        )
        t2_feats, t2_coords = self._load_slat(
            self.trellis2_global_slat_dir / f"{asset_id}.npz",
            self.trellis2_slat_normalization,
            self.config.trellis2_slat_norm,
        )
        return {
            "asset_id": asset_id,
            "old_feats": old_feats,
            "old_coords": old_coords,
            "trellis2_feats": t2_feats,
            "trellis2_coords": t2_coords,
        }


class GlobalSLatPairData:
    def __init__(self, config: GlobalSLatPairDataConfig, timer=None):
        self.config = config
        self.dataset = GlobalSLatPairDataset(config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_processes,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, examples):
        return {
            "batch_id": [example["asset_id"] for example in examples],
            "batch_bbox": [torch.zeros(1, 2, 3) for _ in examples],
            "batch_slat_feats": [[example["old_feats"]] for example in examples],
            "batch_slat_coords": [[example["old_coords"]] for example in examples],
            "batch_trellis2_slat_feats": [[example["trellis2_feats"]] for example in examples],
            "batch_trellis2_slat_coords": [[example["trellis2_coords"]] for example in examples],
        }
