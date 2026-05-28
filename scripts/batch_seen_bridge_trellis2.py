import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src/submodule/TRELLIS"))

from src.models import FullPartToTrellis2SLatAdapter, FullPartToTrellis2SLatAdapterConfig
from src.utils import load_model
import trellis.modules.sparse as old_sp


OLD_SLAT_NORMALIZATION = {
    "mean": [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133, -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826],
    "std": [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695, 2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338],
}


def read_ids(csv_path, old_dir, gt_dir, renders_cond_dir, view):
    ids = []
    with Path(csv_path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset_id = row.get("asset_id")
            if asset_id and row.get("skip", "False") == "False":
                if (
                    (Path(old_dir) / f"{asset_id}.npz").exists()
                    and (Path(gt_dir) / f"{asset_id}.npz").exists()
                    and (Path(renders_cond_dir) / asset_id / view).exists()
                ):
                    ids.append(asset_id)
    return ids


def load_npz(path):
    data = np.load(path)
    return torch.from_numpy(data["feats"]).float(), torch.from_numpy(data["coords"]).int()


def normalize(feats, stats):
    mean = torch.tensor(stats["mean"], dtype=feats.dtype)[None]
    std = torch.tensor(stats["std"], dtype=feats.dtype)[None]
    return (feats - mean) / std


def denormalize(feats, stats, device):
    mean = torch.tensor(stats["mean"], dtype=feats.dtype, device=device)[None]
    std = torch.tensor(stats["std"], dtype=feats.dtype, device=device)[None]
    return feats * std + mean


def add_batch(coords):
    if coords.shape[-1] == 4:
        coords = coords.clone()
        coords[:, 0] = 0
        return coords
    return torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype), coords], dim=-1)


def export_textured_glb(out_mesh, output_path, o_voxel, decimation_target, texture_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = out_mesh[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(output_path)


def run_shape_flow_from_init(pipeline, init_slat, cond, flow_model, steps, strength):
    if strength > 0.0:
        random_noise = init_slat.replace(feats=torch.randn_like(init_slat.feats))
        init_slat = init_slat.replace(feats=(1.0 - strength) * init_slat.feats + strength * random_noise.feats)
    sampler_params = {**pipeline.shape_slat_sampler_params, "steps": steps}
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    slat = pipeline.shape_slat_sampler.sample(
        flow_model,
        init_slat,
        **cond,
        **sampler_params,
        verbose=True,
        tqdm_desc="Sampling shape SLat from adapter init",
    ).samples
    if pipeline.low_vram:
        flow_model.cpu()
    return denormalize(slat.feats, pipeline.shape_slat_normalization, slat.device), slat.coords.int()


def main():
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/partversexl/home_sample_1000.csv")
    parser.add_argument("--old-dir", default="dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--gt-dir", default="dataset/partversexl/trellis2_global_shape_latents")
    parser.add_argument("--renders-cond-dir", default="dataset/partversexl/renders_cond")
    parser.add_argument("--view", default="000.png")
    parser.add_argument("--stats-json", default="dataset/partversexl/trellis2_shape_latents_stats.json")
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--output-dir", default="outputs/adapter_seen_bridge_batch10")
    parser.add_argument("--trellis2-root", default="/root/data/TRELLIS.2")
    parser.add_argument("--trellis2-pretrained", default="/root/data/pretrained_models/trellis.2-4B/ckpts")
    parser.add_argument("--config-file", default="pipeline_local.json")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-gt", action="store_true")
    parser.add_argument("--refine-with-shape-flow", action="store_true")
    parser.add_argument("--shape-flow-steps", type=int, default=12)
    parser.add_argument("--shape-init-noise-strength", type=float, default=0.0)
    parser.add_argument("--decimation-target", type=int, default=300000)
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()

    if args.trellis2_root not in sys.path:
        sys.path.insert(0, args.trellis2_root)

    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    with Path(args.stats_json).open("r") as f:
        t2_stats = json.load(f)

    ids = read_ids(args.csv, args.old_dir, args.gt_dir, args.renders_cond_dir, args.view)
    ids = ids[args.start_index: args.start_index + args.num_samples]
    if not ids:
        raise RuntimeError("No valid samples found")

    adapter = FullPartToTrellis2SLatAdapter(FullPartToTrellis2SLatAdapterConfig())
    adapter = load_model(adapter, args.adapter_ckpt, rename_func=None)
    adapter = adapter.to(device).eval()

    print("Loading TRELLIS.2 image-to-3D pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.trellis2_pretrained, config_file=args.config_file)
    pipeline.to(device)
    pipeline.low_vram = True
    shape_model_name = "shape_slat_flow_model_512" if args.resolution == 512 else "shape_slat_flow_model_1024"
    tex_model_name = "tex_slat_flow_model_512" if args.resolution == 512 else "tex_slat_flow_model_1024"

    output_dir = Path(args.output_dir)
    summary = []
    for idx, asset_id in enumerate(ids):
        print(f"=== [{idx + 1}/{len(ids)}] {asset_id} ===")
        old_feats, old_coords = load_npz(Path(args.old_dir) / f"{asset_id}.npz")
        gt_feats_raw, gt_coords = load_npz(Path(args.gt_dir) / f"{asset_id}.npz")
        old_feats = normalize(old_feats, OLD_SLAT_NORMALIZATION)
        gt_feats = normalize(gt_feats_raw, t2_stats)
        old_coords = add_batch(old_coords)
        gt_coords = add_batch(gt_coords)

        old_slat = old_sp.SparseTensor(feats=old_feats.to(device), coords=old_coords.int().to(device))
        with torch.no_grad():
            pred_slat = adapter.old_to_trellis2_slat(old_slat, target_coords=gt_coords.to(device))
            pred_slat.coords[:, 0] = 0

        pred_feats_norm = pred_slat.feats.float()
        gt_feats_device = gt_feats.to(device)
        mse = torch.mean((pred_feats_norm - gt_feats_device) ** 2).item()
        mae = torch.mean(torch.abs(pred_feats_norm - gt_feats_device)).item()
        cosine = torch.nn.functional.cosine_similarity(pred_feats_norm, gt_feats_device, dim=-1).mean().item()

        image_path = Path(args.renders_cond_dir) / asset_id / args.view
        image = Image.open(image_path)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        image = pipeline.preprocess_image(image)
        torch.manual_seed(args.seed)
        cond = pipeline.get_cond([image], args.resolution)

        if args.refine_with_shape_flow:
            adapter_init_slat = SparseTensor(feats=pred_feats_norm, coords=pred_slat.coords.int())
            pred_feats, pred_coords = run_shape_flow_from_init(
                pipeline,
                adapter_init_slat,
                cond,
                pipeline.models[shape_model_name],
                args.shape_flow_steps,
                args.shape_init_noise_strength,
            )
            shape_slat = SparseTensor(feats=pred_feats, coords=pred_coords.int())
        else:
            pred_feats = denormalize(pred_feats_norm, t2_stats, device)
            shape_slat = SparseTensor(feats=pred_feats, coords=pred_slat.coords.int())
        print("Sampling texture SLat...")
        tex_slat = pipeline.sample_tex_slat(cond, pipeline.models[tex_model_name], shape_slat)
        print("Decoding shape + texture latent...")
        out_mesh = pipeline.decode_latent(shape_slat, tex_slat, args.resolution)
        glb_path = output_dir / asset_id / f"pred_textured_{Path(args.view).stem}.glb"
        print(f"Exporting textured GLB: {glb_path}")
        export_textured_glb(out_mesh, glb_path, o_voxel, args.decimation_target, args.texture_size)

        gt_glb_path = None
        if args.export_gt:
            gt_feats_denorm = denormalize(gt_feats_device.float(), t2_stats, device)
            gt_shape_slat = SparseTensor(feats=gt_feats_denorm, coords=gt_coords.to(device).int())
            print("Sampling GT texture SLat...")
            torch.manual_seed(args.seed)
            gt_tex_slat = pipeline.sample_tex_slat(cond, pipeline.models[tex_model_name], gt_shape_slat)
            print("Decoding GT shape + texture latent...")
            gt_out_mesh = pipeline.decode_latent(gt_shape_slat, gt_tex_slat, args.resolution)
            gt_glb_path = output_dir / asset_id / f"gt_textured_{Path(args.view).stem}.glb"
            print(f"Exporting GT textured GLB: {gt_glb_path}")
            export_textured_glb(gt_out_mesh, gt_glb_path, o_voxel, args.decimation_target, args.texture_size)

        item = {
            "asset_id": asset_id,
            "view": args.view,
            "mse_norm": mse,
            "mae_norm": mae,
            "cosine_norm": cosine,
            "old_tokens": int(old_feats.shape[0]),
            "target_tokens": int(gt_feats.shape[0]),
            "condition_image": str(image_path),
            "pred_glb": str(glb_path),
            "gt_glb": str(gt_glb_path) if gt_glb_path is not None else None,
        }
        summary.append(item)
        (output_dir / asset_id).mkdir(parents=True, exist_ok=True)
        with (output_dir / asset_id / "metrics.json").open("w") as f:
            json.dump(item, f, indent=2)
        with (output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        torch.cuda.empty_cache()

    print(f"Done. Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
