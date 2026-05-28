import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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


def normalize(feats, stats):
    mean = torch.tensor(stats["mean"], dtype=feats.dtype)[None]
    std = torch.tensor(stats["std"], dtype=feats.dtype)[None]
    return (feats - mean) / std


def denormalize(feats, stats, device):
    mean = torch.tensor(stats["mean"], dtype=feats.dtype, device=device)[None]
    std = torch.tensor(stats["std"], dtype=feats.dtype, device=device)[None]
    return feats * std + mean


def add_batch(coords):
    coords = coords.int()
    if coords.shape[-1] == 4:
        coords = coords.clone()
        coords[:, 0] = 0
        return coords
    return torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype), coords], dim=-1)


def coords_to_world(coords, resolution):
    xyz = coords[:, -3:].float()
    return (xyz + 0.5) / float(resolution) - 0.5


def bbox_to_grid_bbox(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    return bbox / 2.0


def mask_coords_by_bbox(coords, bbox, resolution, padding):
    pts = coords_to_world(coords, resolution)
    box = torch.from_numpy(bbox_to_grid_bbox(bbox)).to(pts.device, pts.dtype)
    box_min = box[0] - padding
    box_max = box[1] + padding
    return ((pts >= box_min) & (pts <= box_max)).all(dim=-1)


def read_first_asset_id(csv_path):
    with Path(csv_path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("asset_id") and row.get("skip", "False") == "False":
                return row["asset_id"]
    raise RuntimeError("No valid asset_id found")


def load_bboxes(anno_dir, asset_id):
    # 优先检查 demo 格式：anno_dir/asset_id.npy
    npy_path_demo = Path(anno_dir) / f"{asset_id}.npy"
    if npy_path_demo.exists():
        return np.load(npy_path_demo, allow_pickle=False).astype(np.float32)
    
    # 检查 dataset 格式：anno_dir/asset_id/asset_id.npy 或 anno_dir/asset_id/asset_id_info.json
    json_path = Path(anno_dir) / asset_id / f"{asset_id}_info.json"
    npy_path = Path(anno_dir) / asset_id / f"{asset_id}.npy"
    if json_path.exists():
        with json_path.open("r") as f:
            info = json.load(f)
        return np.asarray(info["bboxes"], dtype=np.float32)
    elif npy_path.exists():
        return np.load(npy_path, allow_pickle=False).astype(np.float32)
    else:
        raise FileNotFoundError(f"No bbox file found at {npy_path_demo}, {json_path} or {npy_path}")


def export_textured_glb(out_mesh, output_path, o_voxel, decimation_target, texture_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = out_mesh[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices.detach(),
        faces=mesh.faces.detach(),
        attr_volume=mesh.attrs.detach(),
        coords=mesh.coords.detach(),
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


def decode_latent_safe(pipeline, shape_slat, tex_slat, resolution):
    meshes, subs = pipeline.decode_shape_slat(shape_slat, resolution)
    tex_voxels = pipeline.decode_tex_slat(tex_slat, subs)
    out_mesh = []
    for m, v in zip(meshes, tex_voxels):
        try:
            m.fill_holes()
        except RuntimeError as exc:
            if "CuMesh" not in str(exc) and "fill_holes" not in str(exc) and "invalid configuration argument" not in str(exc):
                raise
            print(f"Skipping mesh hole filling after CuMesh failure: {exc}")
        out_mesh.append(
            MeshWithVoxel(
                m.vertices, m.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=v.coords[:, 1:],
                attrs=v.feats,
                voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                layout=pipeline.pbr_attr_layout,
            )
        )
    return out_mesh


def run_shape_flow_from_init(pipeline, init_slat, cond, flow_model, steps, start_t):
    """Run TRELLIS2 shape flow inference initialized from adapter output as x0 prior.

    init_slat is treated as a clean x0 estimate in TRELLIS2 normalized shape latent space.
    We construct x_t = (1 - t) * x0 + (sigma_min + (1 - sigma_min) * t) * eps, then run
    Euler steps from start_t down to 0. start_t=0 disables refinement (returns init_slat).
    start_t=1.0 reduces to TRELLIS2 native sampling from pure noise (adapter feats ignored,
    only its sparse coords/support remain).
    """
    if start_t <= 0.0:
        return init_slat
    sampler_params = dict(pipeline.shape_slat_sampler_params)
    rescale_t = sampler_params.pop("rescale_t", 1.0)
    verbose = sampler_params.pop("verbose", True)
    sampler_params.pop("steps", None)
    sigma_min = pipeline.shape_slat_sampler.sigma_min
    noise = torch.randn_like(init_slat.feats)
    x_t_feats = (1.0 - start_t) * init_slat.feats + (sigma_min + (1.0 - sigma_min) * start_t) * noise
    sample = init_slat.replace(feats=x_t_feats)
    t_seq = np.linspace(start_t, 0.0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(steps))
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    for t, t_prev in tqdm(t_pairs, desc=f"Refining part shape SLat from t={start_t:.2f}", disable=not verbose):
        out = pipeline.shape_slat_sampler.sample_once(flow_model, sample, t, t_prev, **cond, **sampler_params)
        sample = out.pred_x_prev
    if pipeline.low_vram:
        flow_model.cpu()
    return sample


def main():
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--csv", default="dataset/partversexl/home_sample_1000.csv")
    parser.add_argument("--old-part-dir", default=None)
    parser.add_argument("--old-global-dir", default="dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--use-part-slats", action="store_true", help="Use part-level SLats (part_{i}_slat.npz) from Stage2 output instead of splitting global_slat")
    parser.add_argument("--stage2-output-dir", default=None, help="Stage2 output directory containing part_{i}_slat.npz files")
    parser.add_argument("--anno-dir", default="dataset/partversexl/anno_infos")
    parser.add_argument("--renders-cond-dir", default="dataset/partversexl/renders_cond")
    parser.add_argument("--view", default="000.png")
    parser.add_argument("--stats-json", default="dataset/partversexl/trellis2_shape_latents_stats.json")
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--output-dir", default="outputs/part_bbox_adapter_trellis2")
    parser.add_argument("--trellis2-root", default="/root/data/TRELLIS.2")
    parser.add_argument("--trellis2-pretrained", default="/root/data/pretrained_models/trellis.2-4B/ckpts")
    parser.add_argument("--config-file", default="pipeline_local.json")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024])
    parser.add_argument("--old-resolution", type=int, default=64)
    parser.add_argument("--part-ids", type=int, nargs="*", default=None)
    parser.add_argument("--bbox-padding", type=float, default=0.02)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--max-old-tokens", type=int, default=50000)
    parser.add_argument("--refine-with-shape-flow", action="store_true")
    parser.add_argument("--shape-flow-steps", type=int, default=12)
    parser.add_argument("--shape-init-noise-strength", type=float, default=0.0,
                        help="[deprecated] kept for backward compatibility; use --shape-flow-start-t")
    parser.add_argument("--shape-flow-start-t", type=float, default=None,
                        help="Starting timestep t in [0,1] for TRELLIS2 shape flow inference. "
                             "0.0 disables refinement (use adapter output directly). "
                             "0.2~0.4 light refinement preserving adapter prior. "
                             "0.7~0.9 TRELLIS2 generative dominates. "
                             "1.0 fully trust TRELLIS2 (adapter feats overwritten by noise, only coords kept).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decimation-target", type=int, default=200000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--skip-texture", action="store_true", help="Skip texture sampling and decode, only generate shape mesh")
    args = parser.parse_args()

    if args.trellis2_root not in sys.path:
        sys.path.insert(0, args.trellis2_root)

    from trellis2.modules.sparse import SparseTensor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.representations import MeshWithVoxel
    import o_voxel
    globals()["MeshWithVoxel"] = MeshWithVoxel

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    asset_id = args.asset_id or read_first_asset_id(args.csv)
    output_dir = Path(args.output_dir) / asset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.stats_json).open("r") as f:
        t2_stats = json.load(f)

    bboxes = load_bboxes(args.anno_dir, asset_id)
    if args.old_part_dir is None and not args.use_part_slats:
        old_npz = np.load(Path(args.old_global_dir) / f"{asset_id}.npz")
        old_feats = torch.from_numpy(old_npz["feats"]).float()
        old_coords = add_batch(torch.from_numpy(old_npz["coords"]).int())
        old_feats = normalize(old_feats, OLD_SLAT_NORMALIZATION)

    adapter = FullPartToTrellis2SLatAdapter(FullPartToTrellis2SLatAdapterConfig())
    adapter = load_model(adapter, args.adapter_ckpt, rename_func=None)
    adapter = adapter.to(device).eval()

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.trellis2_pretrained, config_file=args.config_file)
    pipeline.to(device)
    pipeline.low_vram = True
    shape_model_name = "shape_slat_flow_model_512" if args.resolution == 512 else "shape_slat_flow_model_1024"
    tex_model_name = "tex_slat_flow_model_512" if args.resolution == 512 else "tex_slat_flow_model_1024"

    # Load global condition image (used for all parts, following FullPart's approach)
    # 支持 demo 格式：renders_cond_dir/asset_id.png 和 dataset 格式：renders_cond_dir/asset_id/view.png
    image_path_demo = Path(args.renders_cond_dir) / f"{asset_id}.png"
    if image_path_demo.exists():
        image_path = image_path_demo
    else:
        image_path = Path(args.renders_cond_dir) / asset_id / args.view
    image = Image.open(image_path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    image = pipeline.preprocess_image(image)
    torch.manual_seed(args.seed)
    cond = pipeline.get_cond([image], args.resolution)

    selected_part_ids = args.part_ids if args.part_ids is not None and len(args.part_ids) > 0 else list(range(len(bboxes)))
    summary = []
    for part_id in selected_part_ids:
        if args.use_part_slats and args.stage2_output_dir is not None:
            # Load part-level SLat directly from Stage2 output
            part_slat_path = Path(args.stage2_output_dir) / asset_id / f"part_{part_id}_slat.npz"
            if not part_slat_path.exists():
                print(f"Skip part {part_id}: missing {part_slat_path}")
                continue
            part_slat_npz = np.load(part_slat_path)
            part_feats = normalize(torch.from_numpy(part_slat_npz["feats"]).float(), OLD_SLAT_NORMALIZATION)
            part_coords = add_batch(torch.from_numpy(part_slat_npz["coords"]).int())
            token_count = int(part_feats.shape[0])
            if token_count < args.min_tokens:
                print(f"Skip part {part_id}: only {token_count} tokens")
                continue
            part_old_slat = old_sp.SparseTensor(
                feats=part_feats.to(device),
                coords=part_coords.int().to(device),
            )
            target_coords = None
        elif args.old_part_dir is not None:
            old_part_path = Path(args.old_part_dir) / asset_id / f"{part_id}.npz"
            if not old_part_path.exists():
                print(f"Skip part {part_id}: missing {old_part_path}")
                continue
            old_part_npz = np.load(old_part_path)
            part_feats = normalize(torch.from_numpy(old_part_npz["feats"]).float(), OLD_SLAT_NORMALIZATION)
            part_coords = add_batch(torch.from_numpy(old_part_npz["coords"]).int())
            token_count = int(part_feats.shape[0])
            if token_count < args.min_tokens:
                print(f"Skip part {part_id}: only {token_count} tokens")
                continue
            part_old_slat = old_sp.SparseTensor(
                feats=part_feats.to(device),
                coords=part_coords.int().to(device),
            )
            target_coords = None
        else:
            bbox = bboxes[part_id]
            mask = mask_coords_by_bbox(old_coords, bbox, args.old_resolution, args.bbox_padding)
            token_count = int(mask.sum().item())
            if token_count < args.min_tokens:
                print(f"Skip part {part_id}: only {token_count} tokens")
                continue
            if args.max_old_tokens is not None and token_count > args.max_old_tokens:
                selected = torch.nonzero(mask, as_tuple=False).flatten()
                generator = torch.Generator(device=selected.device).manual_seed(args.seed + part_id)
                selected = selected[torch.randperm(selected.shape[0], generator=generator)[:args.max_old_tokens]]
                mask = torch.zeros_like(mask)
                mask[selected] = True
                token_count = int(mask.sum().item())
            part_old_slat = old_sp.SparseTensor(
                feats=old_feats[mask].to(device),
                coords=old_coords[mask].int().to(device),
            )
            target_coords = part_old_slat.coords.int()
            target_coords[:, 0] = 0
        with torch.no_grad():
            pred_slat = adapter.old_to_trellis2_slat(part_old_slat, target_coords=target_coords)
            pred_slat.coords[:, 0] = 0
        pred_feats_norm = pred_slat.feats.float()
        init_shape_slat = SparseTensor(feats=pred_feats_norm, coords=pred_slat.coords.int())
        np.savez(
            output_dir / f"part_{part_id:02d}_adapter_shape_slat_norm.npz",
            feats=pred_feats_norm.detach().cpu().numpy(),
            coords=pred_slat.coords.int().detach().cpu().numpy(),
        )
        if args.refine_with_shape_flow:
            refined = run_shape_flow_from_init(
                pipeline,
                init_shape_slat,
                cond,
                pipeline.models[shape_model_name],
                args.shape_flow_steps,
                args.shape_flow_start_t if args.shape_flow_start_t is not None else args.shape_init_noise_strength,
            )
            shape_feats = denormalize(refined.feats.float(), pipeline.shape_slat_normalization, device)
            shape_slat = SparseTensor(feats=shape_feats, coords=refined.coords.int())
        else:
            shape_feats = denormalize(pred_feats_norm, t2_stats, device)
            shape_slat = SparseTensor(feats=shape_feats, coords=pred_slat.coords.int())
        
        if args.skip_texture:
            print(f"Skipping texture for part {part_id}, decoding shape only")
            meshes, subs = pipeline.decode_shape_slat(shape_slat, args.resolution)
            mesh = meshes[0]
            if mesh.faces.shape[0] == 0:
                print(f"Skip part {part_id}: decoded mesh has 0 faces")
                summary.append({"part_id": part_id, "old_tokens": token_count, "glb": None, "error": "decoded mesh has 0 faces"})
                del mesh, shape_slat, init_shape_slat, pred_slat, part_old_slat
                torch.cuda.empty_cache()
                continue
            glb_path = output_dir / f"part_{part_id:02d}_shape_only.glb"
            try:
                import trimesh
                tmesh = trimesh.Trimesh(vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy())
                tmesh.export(glb_path)
            except Exception as exc:
                print(f"Skip part {part_id}: mesh export failed: {exc}")
                summary.append({"part_id": part_id, "old_tokens": token_count, "glb": None, "error": str(exc)})
                del mesh, shape_slat, init_shape_slat, pred_slat, part_old_slat
                torch.cuda.empty_cache()
                continue
            summary.append({"part_id": part_id, "old_tokens": token_count, "glb": str(glb_path)})
            del mesh, shape_slat, init_shape_slat, pred_slat, part_old_slat
            torch.cuda.empty_cache()
        else:
            print(f"Sampling texture for part {part_id} with {token_count} old tokens")
            tex_slat = pipeline.sample_tex_slat(cond, pipeline.models[tex_model_name], shape_slat)
            out_mesh = decode_latent_safe(pipeline, shape_slat, tex_slat, args.resolution)
            mesh = out_mesh[0]
            if mesh.faces.shape[0] == 0:
                print(f"Skip part {part_id}: decoded mesh has 0 faces")
                summary.append({"part_id": part_id, "old_tokens": token_count, "glb": None, "error": "decoded mesh has 0 faces"})
                del tex_slat, out_mesh, mesh, shape_slat, init_shape_slat, pred_slat, part_old_slat
                torch.cuda.empty_cache()
                continue
            glb_path = output_dir / f"part_{part_id:02d}_adapter_trellis2.glb"
            try:
                export_textured_glb(out_mesh, glb_path, o_voxel, args.decimation_target, args.texture_size)
            except RuntimeError as exc:
                if "CuMesh" not in str(exc) and "invalid configuration argument" not in str(exc):
                    raise
                print(f"Skip part {part_id}: GLB postprocess failed: {exc}")
                summary.append({"part_id": part_id, "old_tokens": token_count, "glb": None, "error": str(exc)})
                del tex_slat, out_mesh, mesh, shape_slat, init_shape_slat, pred_slat, part_old_slat
                torch.cuda.empty_cache()
                continue
            summary.append({"part_id": part_id, "old_tokens": token_count, "glb": str(glb_path)})
            del tex_slat, out_mesh, mesh, shape_slat, init_shape_slat, pred_slat, part_old_slat
            torch.cuda.empty_cache()

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Saved {len(summary)} part GLBs to {output_dir}")


if __name__ == "__main__":
    main()
