from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type

import torch
from diffusers.models.modeling_utils import ModelMixin
from torch import nn

from ...configs.base_config import InstantiateConfig
from ...utils import load_model


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
class FullPartToTrellis2SLatAdapterConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: FullPartToTrellis2SLatAdapter)
    old_channels: int = 8
    trellis2_channels: int = 32
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    old_resolution: int = 64
    trellis2_resolution: int = 32
    coord_fourier_bands: int = 6
    adapter_ckpt_path: Optional[str] = None

    def from_pretrained(self, **kwargs) -> Any:
        adapter = FullPartToTrellis2SLatAdapter(self)
        if self.adapter_ckpt_path is not None:
            adapter = load_model(adapter, self.adapter_ckpt_path, rename_func=None)
        return adapter


class FullPartToTrellis2SLatAdapter(ModelMixin):
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

    def forward(
        self,
        old_feats: torch.Tensor,
        old_coords: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        target_tokens = self.target_coord_proj(target_coord_emb).unsqueeze(0)
        hidden = target_tokens
        for block in self.blocks:
            attn_out, _ = block["cross_attn"](
                block["norm_q"](hidden),
                block["norm_kv"](old_tokens),
                block["norm_kv"](old_tokens),
                need_weights=False,
            )
            hidden = hidden + attn_out
            hidden = hidden + block["ff"](block["norm_ff"](hidden))
        pred_feats = self.out_proj(hidden.squeeze(0))
        return pred_feats, target_coords

    def old_to_trellis2_slat(self, old_slat, target_coords: Optional[torch.Tensor] = None):
        pred_feats, pred_coords = self(old_slat.feats, old_slat.coords, target_coords)
        return old_slat.replace(feats=pred_feats, coords=pred_coords)
