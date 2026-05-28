import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OLD_SLAT_NORMALIZATION = {
    "mean": [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133, -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826],
    "std": [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695, 2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338],
}


@dataclass
class SimpleSLat:
    feats: torch.Tensor
    coords: torch.Tensor

    @property
    def device(self):
        return self.feats.device

    def replace(self, feats: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None):
        return SimpleSLat(
            feats=self.feats if feats is None else feats,
            coords=self.coords if coords is None else coords,
        )


@dataclass
class FullPartToTrellis2SLatAdapterConfig:
    old_channels: int = 8
    trellis2_channels: int = 32
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    old_resolution: int = 64
    trellis2_resolution: int = 32
    coord_fourier_bands: int = 6


class FullPartToTrellis2SLatAdapter(nn.Module):
    def __init__(self, config: FullPartToTrellis2SLatAdapterConfig):
        super().__init__()
        self.config = config
        coord_dim = 3 * (2 * config.coord_fourier_bands + 1)
        self.old_token_proj = nn.Sequential(
            nn.Linear(config.old_channels + coord_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.target_coord_proj = nn.Sequential(
            nn.Linear(coord_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(config.hidden_dim),
                        "norm_kv": nn.LayerNorm(config.hidden_dim),
                        "cross_attn": nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True),
                        "norm_ff": nn.LayerNorm(config.hidden_dim),
                        "ff": nn.Sequential(
                            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                            nn.SiLU(),
                            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                        ),
                    }
                )
                for _ in range(config.num_layers)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.trellis2_channels),
        )

    def _coord_embed(self, coords: torch.Tensor, resolution: int) -> torch.Tensor:
        coords = coords.float() / max(resolution - 1, 1)
        embeds = [coords]
        for i in range(self.config.coord_fourier_bands):
            freq = float(2 ** i)
            embeds.append(torch.sin(coords * freq * torch.pi))
            embeds.append(torch.cos(coords * freq * torch.pi))
        return torch.cat(embeds, dim=-1)

    def remap_coords(self, old_coords: torch.Tensor) -> torch.Tensor:
        spatial = old_coords[:, -3:].float()
        scale = self.config.trellis2_resolution / self.config.old_resolution
        spatial = torch.clamp((spatial * scale).floor().long(), 0, self.config.trellis2_resolution - 1)
        if old_coords.shape[-1] == 4:
            return torch.cat([old_coords[:, :1].long(), spatial], dim=-1)
        return spatial

    def _unique_mean(self, coords: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_coords, inverse = torch.unique(coords.long(), dim=0, return_inverse=True)
        out = torch.zeros(unique_coords.shape[0], feats.shape[-1], device=feats.device, dtype=feats.dtype)
        count = torch.zeros(unique_coords.shape[0], 1, device=feats.device, dtype=feats.dtype)
        out.index_add_(0, inverse, feats)
        count.index_add_(0, inverse, torch.ones(feats.shape[0], 1, device=feats.device, dtype=feats.dtype))
        return unique_coords, out / count.clamp_min(1)

    def forward(self, old_feats: torch.Tensor, old_coords: torch.Tensor, target_coords: Optional[torch.Tensor] = None):
        model_dtype = next(self.parameters()).dtype
        old_feats = old_feats.to(dtype=model_dtype)
        old_spatial = old_coords[:, -3:]
        old_coord_emb = self._coord_embed(old_spatial, self.config.old_resolution).to(old_feats.dtype)
        old_tokens = self.old_token_proj(torch.cat([old_feats, old_coord_emb], dim=-1)).unsqueeze(0)
        if target_coords is None:
            target_coords = self.remap_coords(old_coords)
        else:
            target_coords = target_coords.to(old_coords.device).long()
        target_coords, _ = self._unique_mean(
            target_coords,
            torch.zeros(target_coords.shape[0], 1, device=old_feats.device, dtype=old_feats.dtype),
        )
        target_spatial = target_coords[:, -3:]
        target_coord_emb = self._coord_embed(target_spatial, self.config.trellis2_resolution).to(old_feats.dtype)
        hidden = self.target_coord_proj(target_coord_emb).unsqueeze(0)
        for block in self.blocks:
            attn_out, _ = block["cross_attn"](
                block["norm_q"](hidden),
                block["norm_kv"](old_tokens),
                block["norm_kv"](old_tokens),
                need_weights=False,
            )
            hidden = hidden + attn_out
            hidden = hidden + block["ff"](block["norm_ff"](hidden))
        return self.out_proj(hidden.squeeze(0)), target_coords

    def old_to_trellis2_slat(self, old_slat, target_coords: Optional[torch.Tensor] = None):
        pred_feats, pred_coords = self(old_slat.feats, old_slat.coords, target_coords)
        return old_slat.replace(feats=pred_feats, coords=pred_coords)


def load_adapter(ckpt_path: Path, device: torch.device):
    adapter = FullPartToTrellis2SLatAdapter(FullPartToTrellis2SLatAdapterConfig())
    state_dict = torch.load(ckpt_path, map_location="cpu")
    adapter.load_state_dict(state_dict, strict=True)
    adapter.to(device)
    adapter.eval()
    return adapter


def resolve_adapter_ckpt(path: str) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        if (ckpt_path / "pytorch_model.ckpt").exists():
            return ckpt_path / "pytorch_model.ckpt"
        if (ckpt_path / "slat_adapter" / "pytorch_model.ckpt").exists():
            return ckpt_path / "slat_adapter" / "pytorch_model.ckpt"
    return ckpt_path


def read_ids(csv_path: Path, old_dir: Path, target_dir: Path | None, num_samples: int | None, require_target: bool):
    ids = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset_id = row.get("asset_id")
            if not asset_id or row.get("skip", "False") != "False":
                continue
            if not (old_dir / f"{asset_id}.npz").exists():
                continue
            if require_target and (target_dir is None or not (target_dir / f"{asset_id}.npz").exists()):
                continue
            ids.append(asset_id)
            if num_samples is not None and len(ids) >= num_samples:
                break
    return ids


def load_slat(path: Path, norm: dict | None = None):
    data = np.load(path)
    feats = torch.from_numpy(data["feats"]).float()
    coords = torch.from_numpy(data["coords"]).long()
    if norm is not None:
        mean = torch.tensor(norm["mean"], dtype=feats.dtype)[None]
        std = torch.tensor(norm["std"], dtype=feats.dtype)[None]
        feats = (feats - mean) / std
    return feats, coords


def maybe_load_json(path: str | None):
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--dataset-csv", default="dataset/partversexl/home_sample_1000.csv")
    parser.add_argument("--old-global-slat-dir", default="dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--trellis2-global-slat-dir", default="dataset/partversexl/trellis2_global_shape_latents")
    parser.add_argument("--trellis2-stats-json", default="dataset/partversexl/trellis2_shape_latents_stats.json")
    parser.add_argument("--output-dir", default="sample_results/global_adapter_preds")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-target-coords", action="store_true")
    parser.add_argument("--no-old-slat-norm", action="store_true")
    parser.add_argument("--denorm-output", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    old_dir = Path(args.old_global_slat_dir)
    target_dir = Path(args.trellis2_global_slat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(resolve_adapter_ckpt(args.adapter_ckpt), device)

    old_norm = None if args.no_old_slat_norm else OLD_SLAT_NORMALIZATION
    target_norm = maybe_load_json(args.trellis2_stats_json)
    ids = read_ids(Path(args.dataset_csv), old_dir, target_dir, args.num_samples, args.use_target_coords)
    if not ids:
        raise RuntimeError("No valid asset ids found for export.")

    metrics = []
    with torch.no_grad():
        for asset_id in ids:
            old_feats, old_coords = load_slat(old_dir / f"{asset_id}.npz", old_norm)
            old_coords = torch.cat([torch.zeros(old_coords.shape[0], 1, dtype=old_coords.dtype), old_coords], dim=-1)
            old_slat = SimpleSLat(feats=old_feats.to(device), coords=old_coords.to(device))

            target_coords = None
            target_feats = None
            if args.use_target_coords:
                target_feats, target_coords = load_slat(target_dir / f"{asset_id}.npz", target_norm)
                target_coords = torch.cat([torch.zeros(target_coords.shape[0], 1, dtype=target_coords.dtype), target_coords], dim=-1).to(device)
                target_feats = target_feats.to(device)

            pred_slat = adapter.old_to_trellis2_slat(old_slat, target_coords=target_coords)
            pred_feats = pred_slat.feats.float()
            pred_coords = pred_slat.coords.long()

            loss = None
            if target_feats is not None and pred_feats.shape[0] == target_feats.shape[0]:
                loss = ((pred_feats - target_feats.float()) ** 2).mean().item()

            if args.denorm_output:
                if target_norm is None:
                    raise RuntimeError("--denorm-output requires a valid --trellis2-stats-json")
                mean = torch.tensor(target_norm["mean"], dtype=pred_feats.dtype, device=pred_feats.device)[None]
                std = torch.tensor(target_norm["std"], dtype=pred_feats.dtype, device=pred_feats.device)[None]
                pred_feats = pred_feats * std + mean

            np.savez_compressed(
                output_dir / f"{asset_id}.npz",
                feats=pred_feats.detach().cpu().numpy().astype(np.float32),
                coords=pred_coords[:, -3:].detach().cpu().numpy().astype(np.int32),
            )
            metrics.append({
                "asset_id": asset_id,
                "num_pred_tokens": int(pred_feats.shape[0]),
                "loss_to_target_norm": loss,
                "pred_mean": float(pred_feats.mean().item()),
                "pred_std": float(pred_feats.std().item()),
            })
            print(f"exported {asset_id}: tokens={pred_feats.shape[0]}, loss={loss}")

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {len(metrics)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
