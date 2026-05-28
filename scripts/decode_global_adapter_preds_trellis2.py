import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def write_ply(path: Path, vertices: torch.Tensor, faces: torch.Tensor):
    vertices = vertices.detach().float().cpu().numpy()
    faces = faces.detach().int().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def load_pred_npz(path: Path, device: torch.device, denorm_stats: dict | None):
    data = np.load(path)
    feats = torch.from_numpy(data["feats"]).float().to(device)
    coords = torch.from_numpy(data["coords"]).int().to(device)
    if coords.shape[-1] == 3:
        coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=device), coords], dim=-1)
    if denorm_stats is not None:
        mean = torch.tensor(denorm_stats["mean"], dtype=feats.dtype, device=device)[None]
        std = torch.tensor(denorm_stats["std"], dtype=feats.dtype, device=device)[None]
        feats = feats * std + mean
    return feats, coords


def export_glb(path: Path, mesh, o_voxel, trimesh):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all(hasattr(mesh, attr) for attr in ("attrs", "coords", "layout", "voxel_size")):
        simple_mesh = trimesh.Trimesh(
            vertices=mesh.vertices.detach().float().cpu().numpy(),
            faces=mesh.faces.detach().int().cpu().numpy(),
            process=False,
        )
        simple_mesh.export(path)
        return
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(path)


def export_video(path: Path, mesh, envmap, render_utils, imageio, fps: int, num_frames: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    render_kwargs = {"envmap": envmap} if all(hasattr(mesh, attr) for attr in ("attrs", "coords", "layout", "voxel_size")) else {}
    result = render_utils.render_video(mesh, num_frames=num_frames, **render_kwargs)
    if "shaded" in result:
        video = render_utils.make_pbr_vis_frames(result)
    else:
        frame_key = next((key for key in ("color", "normal", "depth", "mask") if key in result), next(iter(result)))
        video = result[frame_key]
    imageio.mimsave(path, video, fps=fps)


def load_glb_as_mesh(path: Path, Mesh, trimesh, torch, device: torch.device):
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise RuntimeError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    vertices = torch.from_numpy(np.asarray(loaded.vertices)).float().to(device)
    faces = torch.from_numpy(np.asarray(loaded.faces)).int().to(device)
    return Mesh(vertices, faces)


def maybe_load_json(path: str | None):
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r") as f:
        return json.load(f)


def main():
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gt-dir", default="/root/data/fullpart-main/dataset/partversexl/trellis2_global_shape_latents")
    parser.add_argument("--normalized-glb-dir", default="/root/data/fullpart-main/dataset/partversexl/normalized_glbs")
    parser.add_argument("--trellis2-root", default="/root/data/TRELLIS.2")
    parser.add_argument("--trellis2-pretrained", default="/root/data/pretrained_models/trellis.2-4B/ckpts")
    parser.add_argument("--config-file", default="pipeline_local.json")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--denorm-input", action="store_true")
    parser.add_argument("--trellis2-stats-json", default="dataset/partversexl/trellis2_shape_latents_stats.json")
    parser.add_argument("--conv-backend", default=None, choices=[None, "none", "spconv", "torchsparse", "flex_gemm"])
    parser.add_argument("--export-glb", action="store_true")
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--video-frames", type=int, default=120)
    parser.add_argument("--hdri-path", default="/root/data/TRELLIS.2/assets/hdri/forest.exr")
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--skip-normalized-glb", action="store_true")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else pred_dir / "decoded_shape_ply"
    gt_dir = Path(args.gt_dir)
    normalized_glb_dir = Path(args.normalized_glb_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.trellis2_root not in sys.path:
        sys.path.insert(0, args.trellis2_root)

    from trellis2.modules.sparse import SparseTensor, config as sparse_config
    from trellis2.pipelines.base import Pipeline
    from trellis2.representations import Mesh
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    import cv2
    import imageio
    import o_voxel
    import trimesh

    if args.conv_backend is not None:
        sparse_config.set_conv_backend(args.conv_backend)

    denorm_stats = maybe_load_json(args.trellis2_stats_json) if args.denorm_input else None
    envmap = None
    if args.export_video:
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread(args.hdri_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device=device,
        ))

    class ShapeDecodePipeline(Pipeline):
        model_names_to_load = ["shape_slat_decoder"]

        def __init__(self, models=None):
            super().__init__(models)
            self._device = torch.device("cpu")
            self.low_vram = True

        def to(self, device):
            super().to(device)
            self._device = torch.device(device)

        @torch.no_grad()
        def decode_shape_slat(self, slat, resolution):
            self.models["shape_slat_decoder"].set_resolution(resolution)
            if self.low_vram:
                self.models["shape_slat_decoder"].to(self.device)
                self.models["shape_slat_decoder"].low_vram = True
            ret = self.models["shape_slat_decoder"](slat, return_subs=True)
            if self.low_vram:
                self.models["shape_slat_decoder"].cpu()
                self.models["shape_slat_decoder"].low_vram = False
            return ret

    print("Loading TRELLIS.2 shape decoder...")
    pipeline = ShapeDecodePipeline.from_pretrained(args.trellis2_pretrained, config_file=args.config_file)
    pipeline.to(device)
    pipeline.low_vram = True

    pred_files = sorted(pred_dir.glob(args.pattern))
    pred_files = [p for p in pred_files if p.name != "metrics.json"]
    if args.num_samples is not None:
        pred_files = pred_files[: args.num_samples]
    if not pred_files:
        raise RuntimeError(f"No prediction npz files found in {pred_dir}")

    def decode_and_export(npz_path: Path, kind: str):
        feats, coords = load_pred_npz(npz_path, device, denorm_stats)
        slat = SparseTensor(feats=feats, coords=coords)
        meshes, subs = pipeline.decode_shape_slat(slat, args.resolution)
        if not meshes:
            return {
                "kind": kind,
                "asset_id": npz_path.stem,
                "status": "no_mesh",
                "num_tokens": int(feats.shape[0]),
            }
        mesh = meshes[0]
        stem = f"{npz_path.stem}_{kind}"
        ply_path = output_dir / "ply" / f"{stem}.ply"
        write_ply(ply_path, mesh.vertices, mesh.faces)
        item = {
            "kind": kind,
            "asset_id": npz_path.stem,
            "status": "ok",
            "ply": str(ply_path),
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
            "num_tokens": int(feats.shape[0]),
        }
        if args.export_glb:
            glb_path = output_dir / "glb" / f"{stem}.glb"
            export_glb(glb_path, mesh, o_voxel, trimesh)
            item["glb"] = str(glb_path)
        if args.export_video:
            video_path = output_dir / "video" / f"{stem}.mp4"
            export_video(video_path, mesh, envmap, render_utils, imageio, args.video_fps, args.video_frames)
            item["video"] = str(video_path)
        print(f"Saved {kind} {npz_path.stem}: vertices={mesh.vertices.shape[0]} faces={mesh.faces.shape[0]}")
        return item

    def export_normalized_glb(asset_id: str):
        glb_path = normalized_glb_dir / f"{asset_id}.glb"
        if not glb_path.exists():
            print(f"Normalized GLB not found for {asset_id}: {glb_path}")
            return {
                "kind": "gt",
                "asset_id": asset_id,
                "status": "missing",
                "path": str(glb_path),
            }
        item = {
            "kind": "gt",
            "asset_id": asset_id,
            "status": "ok",
            "glb": str(glb_path),
        }
        if args.export_video:
            mesh = load_glb_as_mesh(glb_path, Mesh, trimesh, torch, device)
            video_path = output_dir / "video" / f"{asset_id}_gt.mp4"
            export_video(video_path, mesh, envmap, render_utils, imageio, args.video_fps, args.video_frames)
            item["video"] = str(video_path)
            item["num_vertices"] = int(mesh.vertices.shape[0])
            item["num_faces"] = int(mesh.faces.shape[0])
        print(f"Saved gt {asset_id}: {glb_path}")
        return item

    summary = []
    with torch.no_grad():
        for pred_path in pred_files:
            if not args.skip_normalized_glb:
                print(f"Rendering gt {pred_path.stem}.glb...")
                summary.append(export_normalized_glb(pred_path.stem))
            print(f"Decoding t2 {pred_path.name}...")
            summary.append(decode_and_export(pred_path, "t2"))
            if not args.skip_gt:
                gt_path = gt_dir / pred_path.name
                if gt_path.exists():
                    print(f"Decoding t1 {gt_path.name}...")
                    summary.append(decode_and_export(gt_path, "t1"))
                else:
                    print(f"T1 latent not found for {pred_path.name}: {gt_path}")
                    summary.append({
                        "kind": "t1",
                        "asset_id": pred_path.stem,
                        "status": "missing",
                        "path": str(gt_path),
                    })
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "decode_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Saved summary to {output_dir / 'decode_summary.json'}")


if __name__ == "__main__":
    main()
