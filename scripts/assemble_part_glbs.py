"""Assemble per-part GLBs into a single GLB in the global coordinate system.

Each per-part GLB produced by TRELLIS2 lives in the normalized cube
[-0.5, 0.5]^3 (matching the decoder aabb). To place it back into the global
frame we need the part's original ``(center, max_edge)``.

FullPart's per-part normalization (see ``src/data/part_data.py::voxelize_part``):
    center    = (bbox_min + bbox_max) / 2
    max_edge  = (bbox_max - bbox_min).max()        # uniform scalar
    local     = (vertices - center) / max_edge     # -> [-0.5, 0.5]
Inverse:
    global    = local * max_edge + center

NOTE: the per-part GLB *file index* used by the latent encoder
(``/mjr/textured_part_glbs/{asset_id}/{i}.glb`` and the npz files) does
**NOT** match the ``bboxes`` array order in ``{asset_id}_info.json``.
The most reliable way to recover ``(center, max_edge)`` for the i-th
part is to read the source per-part GLB at index ``i`` directly and
take its own AABB (those files are already in the global frame).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import trimesh


PART_FNAME_RE = re.compile(r"part_(\d+)_.*\.glb$")


def load_mesh_concat(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    meshes = []
    if isinstance(loaded, trimesh.Scene):
        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                meshes.append(g)
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def part_world_transform(source_glb: Path):
    """Return (max_edge, center) of the source part GLB (already in global frame)."""
    mesh = load_mesh_concat(source_glb)
    if mesh is None:
        raise RuntimeError(f"Empty mesh: {source_glb}")
    v = np.asarray(mesh.vertices)
    vmin = v.min(axis=0).astype(np.float64)
    vmax = v.max(axis=0).astype(np.float64)
    max_edge = float((vmax - vmin).max())
    center = 0.5 * (vmin + vmax)
    return max_edge, center


def transform_part_to_world(mesh: trimesh.Trimesh, bbox: np.ndarray) -> trimesh.Trimesh:
    """Transform mesh from normalized cube [-0.5, 0.5] to world coordinates using bounding box.
    
    This matches the logic in trellis.utils.postprocessing_utils.to_glb when box parameter is provided:
        scale = (box[1] - box[0]).max()
        center = box.mean(axis=0)
        vertices = vertices * scale + center
    """
    scale = (bbox[1] - bbox[0]).max()
    center = bbox.mean(axis=0)
    vertices = mesh.vertices * scale + center
    out = mesh.copy()
    out.vertices = vertices
    return out


def collect_part_glbs(part_dir: Path):
    parts = []
    for p in sorted(part_dir.glob("part_*.glb")):
        m = PART_FNAME_RE.search(p.name)
        if m is None:
            continue
        parts.append((int(m.group(1)), p))
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--part-dir", required=True,
                        help="Directory containing part_XX_*.glb files (TRELLIS2 outputs).")
    parser.add_argument("--source-part-dir", default="/mjr/textured_part_glbs",
                        help="Root containing original per-part GLBs in global frame "
                             "(structure: {root}/{asset_id}/{part_id}.glb). Used to derive "
                             "per-part (max_edge, center) so assembled parts match the original layout.")
    parser.add_argument("--anno-dir", default="dataset/partversexl/anno_infos",
                        help="Directory containing annotation JSON files with bounding boxes. "
                             "Used as fallback when source-part-dir is not available.")
    parser.add_argument("--output", default=None,
                        help="Output GLB path. Defaults to {part_dir}/assembled.glb")
    args = parser.parse_args()

    part_dir = Path(args.part_dir)
    source_root = Path(args.source_part_dir) / args.asset_id
    anno_dir = Path(args.anno_dir) / args.asset_id
    output_path = Path(args.output) if args.output else part_dir / "assembled.glb"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = collect_part_glbs(part_dir)
    if not parts:
        raise RuntimeError(f"No part_XX_*.glb files found under {part_dir}")

    # Load bounding boxes from annotation JSON
    bboxes = None
    if anno_dir.exists():
        info_path = anno_dir / f"{args.asset_id}_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
            bboxes = np.asarray(info["bboxes"], dtype=np.float32)
            print(f"Loaded {len(bboxes)} bounding boxes from {info_path}")

    scene = trimesh.Scene()
    used = []
    for part_id, glb_path in parts:
        # Get bounding box for transformation
        if bboxes is not None and part_id < len(bboxes):
            bbox = bboxes[part_id]
            print(f"Using bbox for part {part_id}: bbox={bbox.round(4).tolist()}")
        else:
            print(f"Skip {glb_path.name}: no bbox available")
            continue
        mesh = load_mesh_concat(glb_path)
        if mesh is None:
            print(f"Skip {glb_path.name}: empty mesh")
            continue
        transformed = transform_part_to_world(mesh, bbox)
        scene.add_geometry(transformed, node_name=f"part_{part_id:02d}")
        used.append((part_id, glb_path.name, bbox))

    if len(scene.geometry) == 0:
        raise RuntimeError("No parts assembled")

    scene.export(output_path)
    print(f"Assembled {len(used)} parts -> {output_path}")
    for part_id, name, bbox in used:
        scale = (bbox[1] - bbox[0]).max()
        center = bbox.mean(axis=0)
        print(f"  part {part_id:02d} <- {name}  scale={scale:.4f} center={center.round(4).tolist()}")


if __name__ == "__main__":
    main()
