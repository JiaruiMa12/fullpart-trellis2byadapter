import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from ..configs.config_utils import to_immutable_dict
from ..models import FullPartToTrellis2SLatAdapter, FullPartToTrellis2SLatAdapterConfig, VoxelTokenizerConfigStage2
from ..models.autoencoders.slat_adapter_trellis2 import SimpleSLat
from ..models.transformers import Transformer2DModelConfig
from .base_pipeline import EDMTrainConfig, PipelineMixin
from .jointdit_single_3d_pipeline_stage2 import JointDiTSingle3DPipelineStage2, JointDiTSingle3DPipelineConfigStage2
from .t2v_pipeline_pixart_alpha import T2VPixArtAlphaPipeline


@dataclass
class JointDiTSingle3DPipelineConfigStage2Trellis2(JointDiTSingle3DPipelineConfigStage2):
    _target: Type = field(default_factory=lambda: JointDiTSingle3DPipelineStage2Trellis2)
    voxel_vae_config: VoxelTokenizerConfigStage2 = VoxelTokenizerConfigStage2()
    slat_adapter_config: FullPartToTrellis2SLatAdapterConfig = FullPartToTrellis2SLatAdapterConfig()
    trellis2_root: str = "/root/data/TRELLIS.2"
    trellis2_pretrained: str = "microsoft/TRELLIS.2-4B"
    trellis2_pipeline_config_file: str = "pipeline.json"
    trellis2_shape_resolution: int = 512
    trellis2_flow_init_strength: float = 0.5
    use_trellis2_flow_refine: bool = False
    call: Dict[str, Any] = to_immutable_dict(
        {
            "num_inference_steps": 25,
            "guidance_scale": 5.0,
        }
    )

    def from_pretrained(self, **kwargs) -> Any:
        pipeline = super().from_pretrained(**kwargs)
        if self.slat_adapter_config is not None and getattr(pipeline, "slat_adapter", None) is None:
            pipeline.register_modules(slat_adapter=self.slat_adapter_config.from_pretrained())
        return pipeline


class JointDiTSingle3DPipelineStage2Trellis2(JointDiTSingle3DPipelineStage2):
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: T5Tokenizer = None,
        text_encoder: T5EncoderModel = None,
        vae: AutoencoderKL = None,
        transformer: Transformer2DModel = None,
        scheduler: DPMSolverMultistepScheduler = None,
        clip_g_model=None,
        clip_l_model: CLIPTextModel = None,
        tokenizer_clip: CLIPTokenizer = None,
        voxel_vae=None,
        slat_adapter: FullPartToTrellis2SLatAdapter = None,
    ):
        if transformer is None:
            T2VPixArtAlphaPipeline.__init__(
                self,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                clip_g_model=clip_g_model,
                clip_l_model=clip_l_model,
                tokenizer_clip=tokenizer_clip,
            )
            self.register_modules(voxel_vae=voxel_vae, slat_adapter=slat_adapter)
        else:
            super().__init__(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                clip_g_model=clip_g_model,
                clip_l_model=clip_l_model,
                tokenizer_clip=tokenizer_clip,
                voxel_vae=voxel_vae,
            )
            self.register_modules(slat_adapter=slat_adapter)
        self._trellis2_pipeline = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def build_slat(self, batch_slat_feats, batch_slat_coords, batch_size: int = 1):
        slat_coords_cat = []
        slat_feats_cat = []
        for bs_id, slat_feats_list in enumerate(batch_slat_feats):
            for i, slat_feats in enumerate(slat_feats_list):
                slat_coords = batch_slat_coords[bs_id][i]
                part_batch_id = torch.ones(
                    slat_coords.shape[0], 1, dtype=slat_coords.dtype, device=slat_coords.device
                ) * bs_id
                slat_coords = torch.cat([part_batch_id, slat_coords], dim=-1)
                slat_coords_cat.append(slat_coords)
                slat_feats_cat.append(slat_feats)
        slat_coords_cat = torch.cat(slat_coords_cat, dim=0).long().cuda()
        slat_feats_cat = torch.cat(slat_feats_cat, dim=0).float().cuda()
        return SimpleSLat(feats=slat_feats_cat, coords=slat_coords_cat)

    def forward(self, batch):
        old_slat = self.build_slat(batch["batch_slat_feats"], batch["batch_slat_coords"], batch_size=len(batch["batch_bbox"]))
        target_feats = batch.get("batch_trellis2_slat_feats")
        target_coords = batch.get("batch_trellis2_slat_coords")
        if target_feats is None or target_coords is None:
            pred_trellis2_slat = self.slat_adapter.old_to_trellis2_slat(old_slat)
            return {
                "total_loss": pred_trellis2_slat.feats.sum() * 0.0,
                "loss_adapter_feat": pred_trellis2_slat.feats.sum() * 0.0,
            }
        target_slat = self.build_slat(target_feats, target_coords, batch_size=len(batch["batch_bbox"]))
        pred_trellis2_slat = self.slat_adapter.old_to_trellis2_slat(old_slat, target_coords=target_slat.coords)
        pred_feats, matched_target_feats = self.match_target_feats(pred_trellis2_slat, target_slat)
        loss = ((pred_feats.float() - matched_target_feats.float()) ** 2).mean()
        return {"total_loss": loss, "loss_adapter_feat": loss}

    def match_target_feats(self, pred_slat, target_slat):
        pred_map = {tuple(coord.tolist()): i for i, coord in enumerate(pred_slat.coords.detach().cpu())}
        target_indices = []
        pred_indices = []
        for target_i, coord in enumerate(target_slat.coords.detach().cpu()):
            pred_i = pred_map.get(tuple(coord.tolist()))
            if pred_i is not None:
                pred_indices.append(pred_i)
                target_indices.append(target_i)
        if len(pred_indices) == 0:
            pred_indices = torch.arange(min(pred_slat.feats.shape[0], target_slat.feats.shape[0]), device=pred_slat.feats.device)
            target_indices = torch.arange(pred_indices.shape[0], device=target_slat.feats.device)
        else:
            pred_indices = torch.tensor(pred_indices, device=pred_slat.feats.device)
            target_indices = torch.tensor(target_indices, device=target_slat.feats.device)
        return pred_slat.feats[pred_indices], target_slat.feats[target_indices]

    def load_trellis2_pipeline(self):
        if self._trellis2_pipeline is not None:
            return self._trellis2_pipeline
        if self.pipeline_config.trellis2_root not in sys.path:
            sys.path.insert(0, self.pipeline_config.trellis2_root)
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            self.pipeline_config.trellis2_pretrained,
            config_file=self.pipeline_config.trellis2_pipeline_config_file,
        )
        pipeline.to(self._execution_device)
        self._trellis2_pipeline = pipeline
        return pipeline

    @torch.no_grad()
    def decode_with_trellis2_shape_decoder(self, trellis2_slat, resolution: Optional[int] = None):
        pipeline = self.load_trellis2_pipeline()
        resolution = resolution or self.pipeline_config.trellis2_shape_resolution
        return pipeline.decode_shape_slat(trellis2_slat, resolution)

    @torch.no_grad()
    def sample_shape_slat_from_adapter_init(self, cond: dict, flow_model, init_slat, sampler_params: Optional[dict] = None):
        pipeline = self.load_trellis2_pipeline()
        sampler_params = sampler_params or {}
        std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(init_slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(init_slat.device)
        init_norm = init_slat.replace((init_slat.feats - mean) / std)
        strength = self.pipeline_config.trellis2_flow_init_strength
        noise = init_norm.replace(torch.randn_like(init_norm.feats))
        x_t = init_norm.replace((1 - strength) * init_norm.feats + strength * noise.feats)
        params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model,
            x_t,
            **cond,
            **params,
            verbose=True,
            tqdm_desc="Refining shape SLat from FullPart adapter",
        ).samples
        if pipeline.low_vram:
            flow_model.cpu()
        return slat.replace(slat.feats * std + mean)
