"""
FullPart 到 TRELLIS2 推理脚本

支持输入格式：
- 图像：image.png（如 assets/demo_examples/robot.png）
- 边界框：bbox.npy（自动从同目录加载，如 assets/demo_examples/robot.npy）

推理流程：
1. FullPart Stage1：生成粗粒度 part 网格（voxel grids）
2. FullPart Stage2：生成全局稀疏 latent（global slat）并保存独立的 part SLats
3. Adapter 转换：FullPart latent -> TRELLIS2 latent
4. TRELLIS2 shape flow refinement（形状细化）
5. TRELLIS2 texture decode（纹理解码）
6. Assembly：组装 part GLBs 为最终 3D 模型

使用示例：
    python scripts/inference_adapter.py \\
        --raw-path assets/demo_examples/toy_gun.png \\
        --raw-box assets/demo_examples/toy_gun.npy \\
        --stage1.transformer-ckpt /path/to/s1.ckpt \\
        --stage2.transformer-ckpt /path/to/s2.ckpt \\
        --adapter-ckpt /path/to/adapter.ckpt \\
        --output-dir outputs/demo_inference
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# 设置环境变量避免 SIGFPE 崩溃
# 注意：只设置 SPCONV_ALGO 为 native，保持 attention backend 为默认（flash_attn/sdpa）
# 以避免使用 naive attention 导致的 OOM 问题
os.environ["SPCONV_ALGO"] = "native"

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 添加 TRELLIS 子模块到路径
TRELLIS_PATH = REPO_ROOT / "src" / "submodule" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_PATH))

import torch
import tyro
import numpy as np
from PIL import Image
from torchvision import transforms

from src.configs.train_configs.personal_configs_part import personal_configs_part
from src.configs.train_configs.personal_configs_part_stage2 import (
    personal_configs_part_s2,
)
from src.data import DataConfig_3DMaster_Part
from src.data.part_data import preprocess_image
from src.utils import CONSOLE


@dataclass
class StageSettings:
    """Stage pipeline 配置覆盖"""
    config_key: str = "3dmaster_part"  # 配置键名
    transformer_ckpt: Optional[str] = None  # Transformer checkpoint 路径
    num_inference_steps: Optional[int] = None  # 推理步数
    guidance_scale: Optional[float] = None  # 引导系数


@dataclass
class Args:
    """命令行参数（通过 tyro 解析）"""
    # 输入参数（格式：image.png + bbox.npy）
    raw_path: Path  # 图像文件路径（如 assets/demo_examples/robot.png）
    raw_box: Optional[Path] = None  # bbox 路径（可选，默认使用同目录 .npy）
    
    # Stage 配置
    stage1: StageSettings = field(default_factory=StageSettings)
    stage2: StageSettings = field(
        default_factory=lambda: StageSettings(config_key="3dmaster_part_s2")
    )
    
    # 输出配置
    output_dir: Path = Path("./outputs/demo_inference")  # 输出目录
    stage1_subdir: str = "stage1"  # Stage1 输出子目录
    stage2_subdir: str = "stage2"  # Stage2 输出子目录
    trellis2_subdir: str = "trellis2"  # TRELLIS2 输出子目录
    device: str = "cuda"  # 设备
    
    # 跳过步骤
    skip_stage1: bool = False  # 跳过 Stage1
    skip_stage2: bool = False  # 跳过 Stage2
    skip_stage2_decode: bool = True  # 跳过 Stage2 网格解码，直接使用 slat（推荐）
    skip_adapter: bool = False  # 跳过 Adapter
    skip_trellis2: bool = False  # 跳过 TRELLIS2
    skip_assembly: bool = False  # 跳过 Assembly
    stage2_fp32: bool = False  # Stage2 使用 fp32（不推荐，与 FlashAttention 冲突）
    decode_part_ids: Tuple[int, ...] = ()  # 解码的 part ID 列表（空表示解码所有）
    
    # Adapter 配置
    adapter_ckpt: Optional[str] = None  # Adapter checkpoint 路径（必需）
    stats_json: str = "dataset/partversexl/trellis2_shape_latents_stats.json"  # 统计 JSON
    trellis2_root: str = "/root/data/TRELLIS.2"  # TRELLIS.2 根目录
    trellis2_pretrained: str = "/root/data/pretrained_models/trellis.2-4B/ckpts"  # TRELLIS.2 预训练模型
    config_file: str = "pipeline_local.json"  # 配置文件
    
    # TRELLIS2 配置
    texture_size: int = 512  # 纹理大小（降低以减少内存）
    decimation_target: int = 100000  # 网格简化目标面数（降低以减少内存）
    
    # Assembly 配置
    source_part_dir: str = "/mjr/textured_part_glbs"  # 源 part GLB 目录（不需要）
    
    # 其他配置
    max_old_tokens: int = 1500  # 最大 old tokens 数量（降低以避免 OOM）


COND_IMAGE_NORMALIZE = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)


def _resolve_device(device: str) -> torch.device:
    """解析设备参数"""
    if device == "cuda" and not torch.cuda.is_available():
        CONSOLE.log("[yellow]CUDA not available, falling back to CPU[/yellow]")
        return torch.device("cpu")
    return torch.device(device)


def _instantiate_stage1(
    stage_settings: StageSettings,
    device: torch.device,
) -> Tuple[DataConfig_3DMaster_Part, torch.nn.Module]:
    """实例化 Stage1 pipeline"""
    from src.configs.train_configs.personal_configs_part import (
        personal_configs_part,
    )

    cfg = copy.deepcopy(personal_configs_part[stage_settings.config_key])

    pipeline_cfg = cfg.pipeline
    if stage_settings.transformer_ckpt is not None:
        pipeline_cfg.transformer_config.transformer_ckpt_path = stage_settings.transformer_ckpt
        pipeline_cfg.ckpt_path = stage_settings.transformer_ckpt
    if stage_settings.num_inference_steps is not None:
        pipeline_cfg.call["num_inference_steps"] = stage_settings.num_inference_steps
    if stage_settings.guidance_scale is not None:
        pipeline_cfg.call["guidance_scale"] = stage_settings.guidance_scale

    # 设置 ckpt_path 为 None 以避免 from_pretrained 调用 safety_checker 参数
    # 我们将手动加载 transformer
    original_ckpt_path = pipeline_cfg.ckpt_path
    pipeline_cfg.ckpt_path = None
    
    pipeline = pipeline_cfg.from_pretrained()
    
    # 手动加载 transformer checkpoint
    if original_ckpt_path is not None:
        from src.utils import load_model
        pipeline.transformer = load_model(
            pipeline.transformer, 
            original_ckpt_path, 
            rename_func=pipeline_cfg.transformer_config.get_rename_func() if hasattr(pipeline_cfg.transformer_config, 'get_rename_func') else None
        )
    
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()
    if hasattr(pipeline, "voxel_vae"):
        pipeline.voxel_vae.eval()

    return cfg, pipeline


def _instantiate_stage2(
    stage_settings: StageSettings,
    stage1_output_dir: Path,
    stage2_output_dir: Path,
    device: torch.device,
    fp32: bool = False,
) -> torch.nn.Module:
    """实例化 Stage2 pipeline"""
    from src.configs.train_configs.personal_configs_part_stage2 import (
        personal_configs_part_s2,
    )

    cfg = copy.deepcopy(personal_configs_part_s2[stage_settings.config_key])
    pipeline_cfg = cfg.pipeline
    if stage_settings.transformer_ckpt is not None:
        pipeline_cfg.transformer_config.transformer_ckpt_path = stage_settings.transformer_ckpt
        pipeline_cfg.ckpt_path = stage_settings.transformer_ckpt
    if stage_settings.num_inference_steps is not None:
        pipeline_cfg.call["num_inference_steps"] = stage_settings.num_inference_steps
    if stage_settings.guidance_scale is not None:
        pipeline_cfg.call["guidance_scale"] = stage_settings.guidance_scale

    pipeline_cfg.s1_save_dir = str(stage1_output_dir)
    pipeline_cfg.call["save_dir"] = str(stage2_output_dir)

    # 设置 ckpt_path 为 None 以避免 from_pretrained 调用 safety_checker 参数
    # 我们将手动加载 transformer
    original_ckpt_path = pipeline_cfg.ckpt_path
    pipeline_cfg.ckpt_path = None
    
    pipeline = pipeline_cfg.from_pretrained()
    
    # 手动加载 transformer checkpoint
    if original_ckpt_path is not None:
        from src.utils import load_model
        pipeline.transformer = load_model(
            pipeline.transformer, 
            original_ckpt_path, 
            rename_func=pipeline_cfg.transformer_config.get_rename_func() if hasattr(pipeline_cfg.transformer_config, 'get_rename_func') else None
        )
    
    pipeline = pipeline.to(device)
    if fp32:
        pipeline.transformer.float()
    pipeline.transformer.eval()
    if hasattr(pipeline, "voxel_vae"):
        pipeline.voxel_vae.eval()

    return pipeline


def _to_device_dtype(obj, device, dtype) -> None:
    """递归转换对象到指定设备和数据类型"""
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if obj.dtype == torch.float32:
            if dtype == 'bf16':
                return obj.bfloat16()
            elif dtype == 'fp16':
                return obj.half()
        return obj
    elif isinstance(obj, dict):
        return {k: _to_device_dtype(v, device, dtype) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_device_dtype(v, device, dtype) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_device_dtype(v, device, dtype) for v in obj)
    elif isinstance(obj, set):
        return set(_to_device_dtype(v, device, dtype) for v in obj)
    else:
        return obj


def _load_cond_image(image_path: Path) -> torch.Tensor:
    """加载条件图像并归一化"""
    image = Image.open(image_path)
    if image.mode != "RGBA" or image.size[0] != image.size[1]:
        try:
            image = preprocess_image(image)
        except Exception:
            image = image.convert("RGB").resize((518, 518), Image.Resampling.LANCZOS)
    else:
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    tensor = COND_IMAGE_NORMALIZE(tensor)
    return tensor.unsqueeze(0)


def _load_box_tensor(box_path: Path) -> torch.Tensor:
    """加载边界框（支持 .json 和 .npy 格式）"""
    if box_path.suffix == ".json":
        import json
        with open(box_path, "r") as f:
            info = json.load(f)
        box_array = np.asarray(info["bboxes"], dtype=np.float32)
    else:
        box_array = np.load(box_path, allow_pickle=False)
    if box_array.ndim == 2 and box_array.shape == (2, 3):
        box_array = np.expand_dims(box_array, axis=0)
    if box_array.ndim != 3 or box_array.shape[-2:] != (2, 3):
        raise ValueError(
            f"Expected bounding box array of shape (num_parts, 2, 3), got {box_array.shape}."
        )
    return torch.from_numpy(box_array).float()


def _build_canonical_parts(num_parts: int) -> List[torch.Tensor]:
    """构建 canonical part 坐标（全零）"""
    return [torch.zeros((1, 3), dtype=torch.float32) for _ in range(num_parts)]


def _prepare_raw_batch(image_path: Path, box_path: Path, raw_path: Path) -> Dict[str, List[torch.Tensor]]:
    """准备推理 batch（raw_path 为图像文件，box_path 为 bbox 文件）"""
    raw_path = Path(raw_path)
    image_path = raw_path if image_path is None else image_path
    batch_id = raw_path.stem
    if box_path is None:
        # 使用同目录下的 .npy bbox
        box_path = raw_path.parent / (raw_path.stem + ".npy")
    cond_img = _load_cond_image(image_path)
    box_tensor = _load_box_tensor(box_path)
    num_parts = box_tensor.shape[0]
    parts = _build_canonical_parts(num_parts)
    return {
        "batch_cond_imgs": [cond_img],
        "batch_bbox": [box_tensor],
        "batch_part": [parts],
        "batch_id": [batch_id],
    }


def _stage1_inference(
    pipeline: torch.nn.Module,
    batch: Dict[str, List[torch.Tensor]],
    save_dir: Path,
) -> None:
    """运行 Stage1 推理（生成粗粒度 part 网格）"""
    stage_config = pipeline.pipeline_config
    steps = stage_config.call.get("num_inference_steps", 25)
    guidance = stage_config.call.get("guidance_scale", 5.0)

    pipeline(
        batch_cond_imgs=batch["batch_cond_imgs"],
        batch_bbox=batch["batch_bbox"],
        batch_part=batch["batch_part"],
        batch_id=batch["batch_id"],
        save_dir=str(save_dir),
        num_inference_steps=steps,
        guidance_scale=guidance,
    )


def _stage2_inference(
    pipeline: torch.nn.Module,
    batch: Dict[str, List[torch.Tensor]],
    save_dir: Path,
    decode_part_id: Optional[List[int]] = None
) -> None:
    """运行 Stage2 推理（生成全局 slat 和 part SLats）"""
    stage_config = pipeline.pipeline_config
    steps = stage_config.call.get("num_inference_steps", 25)
    guidance = stage_config.call.get("guidance_scale", 5.0)

    call_kwargs = pipeline.prepare_call_kwargs_i2v_from_s1(pipeline, batch)
    call_kwargs.update(
        {
            "save_dir": str(save_dir),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "decode_part_id": decode_part_id,
        }
    )
    pipeline(**call_kwargs)


def _run_adapter_and_trellis2(
    args: Args,
    sample_id: str,
    stage2_output_dir: Path,
    bbox_path: Path,
    anno_dir: Path,
    image_path: Path,
    output_dir: Path,
) -> None:
    """运行 adapter 转换和 TRELLIS2 解码（通过 subprocess）"""
    import subprocess
    
    part_script = Path(__file__).parent / "part_bbox_adapter_trellis2.py"
    
    # renders_cond_dir 为图像文件所在目录
    renders_cond_dir = image_path.parent
    
    # 构建命令：使用 Stage2 输出的 part SLats，启用 shape flow refinement
    cmd = [
        sys.executable,
        str(part_script),
        "--asset-id", sample_id,
        "--anno-dir", str(anno_dir),
        "--renders-cond-dir", str(renders_cond_dir),
        "--stats-json", args.stats_json,
        "--adapter-ckpt", args.adapter_ckpt,
        "--output-dir", str(output_dir),
        "--trellis2-root", args.trellis2_root,
        "--trellis2-pretrained", args.trellis2_pretrained,
        "--config-file", args.config_file,
        "--device", args.device,
        "--decimation-target", str(args.decimation_target),
        "--texture-size", str(args.texture_size),
        "--use-part-slats",
        "--stage2-output-dir", str(stage2_output_dir),
        "--refine-with-shape-flow",
        "--shape-flow-steps", "12",
        "--shape-flow-start-t", "0.9",
        f"--max-old-tokens={args.max_old_tokens}",
    ]
    
    if len(args.decode_part_ids) > 0:
        cmd += ["--part-ids"] + [str(pid) for pid in args.decode_part_ids]
    
    CONSOLE.log(f"Running adapter + TRELLIS2: {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def _run_assembly(
    args: Args,
    sample_id: str,
    trellis2_output_dir: Path,
    output_path: Path,
) -> None:
    """运行 part GLB 组装（通过 subprocess，可选）"""
    import subprocess
    
    assemble_script = Path(__file__).parent / "assemble_part_glbs.py"
    
    cmd = [
        sys.executable,
        str(assemble_script),
        "--asset-id", sample_id,
        "--part-dir", str(trellis2_output_dir / sample_id),
        "--source-part-dir", args.source_part_dir,
        "--output", str(output_path),
    ]
    
    CONSOLE.log(f"Running assembly: {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def main(args: Args) -> None:
    """主函数"""
    device = _resolve_device(args.device)
    torch.set_grad_enabled(False)

    stage1_output_dir = (args.output_dir / args.stage1_subdir).absolute()
    stage2_output_dir = (args.output_dir / args.stage2_subdir).absolute()
    trellis2_output_dir = (args.output_dir / args.trellis2_subdir).absolute()
    
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_stage2:
        stage2_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_trellis2:
        trellis2_output_dir.mkdir(parents=True, exist_ok=True)

    # 准备 batch
    batch = _to_device_dtype(
        _prepare_raw_batch(None, args.raw_box, args.raw_path), 
        device, 
        'bf16'
    )
    
    sample_id = str(batch["batch_id"][0])
    CONSOLE.log(f"Processing sample: {sample_id}")
    
    # Stage 1: FullPart 粗粒度生成
    if not args.skip_stage1:
        CONSOLE.log("Running FullPart Stage1...")
        data_cfg, stage1_pipeline = _instantiate_stage1(args.stage1, device)
        _stage1_inference(stage1_pipeline, batch, stage1_output_dir)
    else:
        CONSOLE.log("Skipping FullPart Stage1")
    
    # Stage 2: FullPart 细化（通过 subprocess 运行以避免 SIGFPE）
    if not args.skip_stage2:
        CONSOLE.log("Running FullPart Stage2...")
        # 通过 subprocess 运行 Stage2 以避免 SIGFPE
        # 只设置 SPCONV_ALGO 为 native，保持 attention backend 为默认以避免 OOM
        import subprocess
        stage2_env = os.environ.copy()
        stage2_env["SPCONV_ALGO"] = "native"
        
        # 传递参数给 Stage2 subprocess（raw-image 为图像文件）
        stage2_cmd = [
            sys.executable,
            str(REPO_ROOT / "inference.py"),
            "--skip-stage1",
            f"--stage1.transformer-ckpt={args.stage1.transformer_ckpt}",
            f"--stage2.transformer-ckpt={args.stage2.transformer_ckpt}",
            f"--raw-image={args.raw_path}",
        ]
        if args.raw_box:
            stage2_cmd.append(f"--raw-box={args.raw_box}")
        stage2_cmd.append(f"--output-dir={args.output_dir}")
        
        # 如果 skip_stage2_decode 为 True，传递空的 decode_part_ids 以跳过网格解码
        if args.skip_stage2_decode:
            stage2_cmd.append("--decode-part-ids")  # 空列表表示不解码任何 part
        # 过滤空字符串
        stage2_cmd = [x for x in stage2_cmd if x]
        
        CONSOLE.log(f"Running Stage2 as subprocess: {' '.join(stage2_cmd)}")
        CONSOLE.log(f"Stage1 outputs are in: {stage1_output_dir / sample_id}")
        CONSOLE.log(f"Files in Stage1 output: {list((stage1_output_dir / sample_id).glob('*'))}")
        CONSOLE.log(f"Expected Stage1 path by subprocess: {args.output_dir / 'stage1' / sample_id}")
        # 从 REPO_ROOT 运行以确保相对路径正确
        subprocess.run(stage2_cmd, env=stage2_env, check=True, cwd=str(REPO_ROOT))
    else:
        CONSOLE.log("Skipping FullPart Stage2")
    
    # Adapter + TRELLIS2（直接使用 Stage2 的 part SLats）
    if not args.skip_adapter and not args.skip_trellis2:
        if args.adapter_ckpt is None:
            raise ValueError("--adapter-ckpt is required for adapter + TRELLIS2 stage")
        
        CONSOLE.log("Running Adapter + TRELLIS2...")
        # raw_path 是图像文件，bbox 在同目录下
        raw_path = Path(args.raw_path)
        image_dir = raw_path.parent
        bbox_path = args.raw_box if args.raw_box else image_dir / (raw_path.stem + ".npy")
        anno_dir = image_dir
        image_path = raw_path
        
        _run_adapter_and_trellis2(
            args, sample_id, stage2_output_dir, bbox_path, anno_dir, image_path, trellis2_output_dir
        )
    else:
        CONSOLE.log("Skipping Adapter + TRELLIS2")
    
    # Assembly
    if not args.skip_assembly:
        CONSOLE.log("Running assembly...")
        assembled_output = args.output_dir / f"{sample_id}_assembled.glb"
        _run_assembly(args, sample_id, trellis2_output_dir, assembled_output)
        CONSOLE.log(f"Done. Assembled GLB: {assembled_output}")
    else:
        CONSOLE.log("Skipping assembly")
    
    CONSOLE.log(f"Pipeline complete. Output directory: {args.output_dir}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
