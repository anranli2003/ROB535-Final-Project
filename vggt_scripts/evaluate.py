"""Evaluate semantic IoU against ground-truth segmentation maps.

This script expects the predicted semantic maps produced by `my_demo.py` or
`my_demo_full.py` (stored as `semantic_map.npy`/`.npz`) and the ground-truth
color-coded segmentation PNGs. It converts both into categorical indices using a
shared label definition and reports per-class IoU scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

# Category specification copied from the provided reference label table.
CATEGORY_INFO: Sequence[Tuple[str, Tuple[int, int, int]]] = (
    ("Terrain", (210, 0, 200)),
    ("Sky", (90, 200, 255)),
    ("Tree", (0, 199, 0)),
    ("Vegetation", (90, 240, 0)),
    ("Building", (140, 140, 140)),
    ("Road", (100, 60, 100)),
    ("GuardRail", (250, 100, 255)),
    ("TrafficSign", (255, 255, 0)),
    ("TrafficLight", (200, 200, 0)),
    ("Pole", (255, 130, 0)),
    ("Misc", (80, 80, 80)),
    ("Truck", (160, 60, 60)),
    ("Car", (255, 127, 80)),
    ("Van", (0, 139, 139)),
    ("Undefined", (0, 0, 0)),
)

CATEGORY_NAMES: List[str] = [name for name, _ in CATEGORY_INFO]
COLOR_TO_INDEX: Dict[int, int] = {
    (r << 16) | (g << 8) | b: idx for idx, (_, (r, g, b)) in enumerate(CATEGORY_INFO)
}
CATEGORY_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
UNDEFINED_CLASS_NAME = "Undefined"
UNDEFINED_INDEX = CATEGORY_TO_INDEX.get(UNDEFINED_CLASS_NAME)

# Mapping from predicted label IDs (as saved by the VGGT demo scripts) to
# canonical category indices. By default, any label ID not listed here will be
# mapped to "Undefined".
PRED_LABEL_TO_CATEGORY: Dict[int, str] = {
    0: "Undefined",  # background / unlabeled
    1: "Terrain",
    2: "Sky",
    3: "Tree",
    4: "Vegetation",
    5: "Building",
    6: "Road",
    7: "GuardRail",
    8: "TrafficSign",
    9: "TrafficLight",
    10: "Pole",
    11: "Misc",
    12: "Truck",
    13: "Car",
    14: "Van",
}

_shared_category_set = set(PRED_LABEL_TO_CATEGORY.values()) & set(CATEGORY_TO_INDEX.keys())
EVALUATED_CATEGORY_NAMES: List[str] = [
    name for name in CATEGORY_NAMES if name in _shared_category_set and name != UNDEFINED_CLASS_NAME
]
if not EVALUATED_CATEGORY_NAMES:
    EVALUATED_CATEGORY_NAMES = [name for name in CATEGORY_NAMES if name != UNDEFINED_CLASS_NAME]
EVALUATED_CATEGORY_INDICES: List[int] = [CATEGORY_TO_INDEX[name] for name in EVALUATED_CATEGORY_NAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IoU between predictions and ground truth.")
    parser.add_argument(
        "--weather",
        type=str,
        default="clone",
        help="Weather condition subdirectory (used to locate default prediction / GT paths).",
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=None,
        help="Path to predicted semantic map (.npy or .npz). Defaults to outputs_full_labels/{weather}/semantic_map.(npz|npy).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help=(
            "Directory with ground-truth PNGs. Defaults to "
            "/home/yehengz/vggt/gt_seg/scene01/{weather}/frames/classSegmentation/Camera_0"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save IoU summary as JSON (defaults to outputs/{weather}/iou_summary.json).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally limit evaluation to the first N frames shared between prediction and GT.",
    )
    return parser.parse_args()


def resolve_prediction_path(weather: str, supplied_path: Path | None) -> Path:
    if supplied_path is not None:
        return supplied_path

    base = Path("./outputs") / weather
    cand_npz = base / "semantic_map.npz"
    cand_npy = base / "semantic_map.npy"
    if cand_npz.exists():
        return cand_npz
    if cand_npy.exists():
        return cand_npy
    raise FileNotFoundError(f"Could not find semantic map in {base}; provide --pred-path explicitly.")


def resolve_gt_dir(weather: str, supplied_dir: Path | None) -> Path:
    if supplied_dir is not None:
        return supplied_dir
    return Path(f"/home/yehengz/vggt/gt_seg/scene01/{weather}/frames/classSegmentation/Camera_0")


def load_prediction_map(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if "semantic_map" not in data:
            raise KeyError(f"Expected key 'semantic_map' in {path}")
        semantic_map = data["semantic_map"]
    else:
        semantic_map = np.load(path, allow_pickle=False)

    if semantic_map.ndim != 3:
        raise ValueError(f"Expected semantic map with shape (N, H, W); got {semantic_map.shape}")

    return semantic_map.astype(np.int32)


def load_gt_maps(
    gt_dir: Path,
    limit: int | None = None,
    target_shape: Tuple[int, int] | None = None,
) -> Tuple[List[Path], List[np.ndarray]]:
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    image_paths = sorted(p for p in gt_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found under {gt_dir}")

    if limit is not None:
        image_paths = image_paths[:limit]

    gt_maps: List[np.ndarray] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        if target_shape is not None and image.size != (target_shape[1], target_shape[0]):
            image = image.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
        rgb = np.array(image, dtype=np.uint8)
        label_map = rgb_to_class_indices(rgb)
        gt_maps.append(label_map)

    return image_paths, gt_maps


def rgb_to_class_indices(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB segmentation image into class indices using CATEGORY_INFO."""

    flat_codes = (
        (rgb[..., 0].astype(np.int32) << 16)
        | (rgb[..., 1].astype(np.int32) << 8)
        | rgb[..., 2].astype(np.int32)
    )

    undefined_idx = CATEGORY_TO_INDEX.get("Undefined", len(CATEGORY_NAMES) - 1)
    label_map = np.full(rgb.shape[:2], undefined_idx, dtype=np.int16)

    for color_code, class_idx in COLOR_TO_INDEX.items():
        mask = flat_codes == color_code
        if np.any(mask):
            label_map[mask] = class_idx

    return label_map.astype(np.int32)


def remap_prediction_to_categories(pred_map: np.ndarray) -> np.ndarray:
    mapped = np.full(pred_map.shape, CATEGORY_TO_INDEX["Undefined"], dtype=np.int32)
    for label_id, category_name in PRED_LABEL_TO_CATEGORY.items():
        class_idx = CATEGORY_TO_INDEX.get(category_name)
        if class_idx is None:
            continue
        mapped[pred_map == label_id] = class_idx
    return mapped


def accumulate_confusion(pred_maps: Iterable[np.ndarray], gt_maps: Iterable[np.ndarray]) -> np.ndarray:
    num_classes = len(CATEGORY_NAMES)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, gt in zip(pred_maps, gt_maps):
        if pred.shape != gt.shape:
            raise ValueError(
                "Prediction and ground-truth map shapes differ: "
                f"pred {pred.shape}, gt {gt.shape}"
            )
        pred_vec = pred.reshape(-1)
        gt_vec = gt.reshape(-1)
        valid_mask = (
            (gt_vec >= 0)
            & (gt_vec < num_classes)
            & (pred_vec >= 0)
            & (pred_vec < num_classes)
        )
        if UNDEFINED_INDEX is not None:
            valid_mask &= gt_vec != UNDEFINED_INDEX
            valid_mask &= pred_vec != UNDEFINED_INDEX

        combined = gt_vec[valid_mask] * num_classes + pred_vec[valid_mask]
        hist = np.bincount(combined, minlength=num_classes * num_classes)
        confusion += hist.reshape(num_classes, num_classes)

    return confusion


def compute_iou(confusion: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    intersection = np.diag(confusion)
    ground_truth = confusion.sum(axis=1)
    prediction = confusion.sum(axis=0)
    union = ground_truth + prediction - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersection / union
    iou[union == 0] = np.nan
    return iou, intersection, union


def main() -> None:
    args = parse_args()

    pred_path = resolve_prediction_path(args.weather, args.pred_path)
    gt_dir = resolve_gt_dir(args.weather, args.gt_dir)

    pred_maps_raw = load_prediction_map(pred_path)

    num_pred_frames = pred_maps_raw.shape[0]
    frame_limit = num_pred_frames
    if args.max_frames is not None:
        frame_limit = min(frame_limit, args.max_frames)

    target_shape = tuple(pred_maps_raw.shape[1:3])
    gt_paths, gt_maps = load_gt_maps(gt_dir, limit=frame_limit, target_shape=target_shape)

    frame_count = min(frame_limit, len(gt_maps), num_pred_frames)
    if frame_count == 0:
        raise ValueError("No frames available for evaluation after alignment.")

    pred_maps_raw = pred_maps_raw[:frame_count]
    gt_maps = gt_maps[:frame_count]
    pred_maps = remap_prediction_to_categories(pred_maps_raw)

    if pred_maps.shape[0] != len(gt_maps):
        raise ValueError(
            "Mismatch between number of predicted frames and ground-truth frames after alignment: "
            f"{pred_maps.shape[0]} vs {len(gt_maps)}"
        )

    confusion = accumulate_confusion(pred_maps, gt_maps)
    iou, intersection, union = compute_iou(confusion)

    eval_indices = np.array(EVALUATED_CATEGORY_INDICES, dtype=int)
    eval_names = EVALUATED_CATEGORY_NAMES
    selected_iou = iou[eval_indices]
    selected_intersection = intersection[eval_indices]
    selected_union = union[eval_indices]

    miou = np.nanmean(selected_iou)

    print("Evaluation summary")
    print(f"Weather: {args.weather}")
    print(f"Prediction file: {pred_path}")
    print(f"Ground-truth directory: {gt_dir}")
    print(f"Frames evaluated: {frame_count}")
    print()
    print("Evaluated categories:")
    print(", ".join(eval_names) if eval_names else "(none)")
    if np.isnan(miou):
        print("Mean IoU: N/A (no overlap in evaluated categories)")
    else:
        print(f"Mean IoU over shared categories: {miou:.4f}")
    print("Per-class IoU (shared categories only):")

    for name, idx, value, inter, uni in zip(
        eval_names,
        eval_indices,
        selected_iou,
        selected_intersection,
        selected_union,
    ):
        display = "N/A" if np.isnan(value) else f"{value:.4f}"
        print(
            f"  {CATEGORY_TO_INDEX[name]:2d} | {name:12s} | IoU: {display:>7} | Intersection: {int(inter)} | Union: {int(uni)}"
        )

    summary = {
        "weather": args.weather,
        "prediction_path": str(pred_path),
        "gt_dir": str(gt_dir),
    "frames": int(frame_count),
        "miou": float(miou) if not np.isnan(miou) else None,
        "categories": [
            {
                "name": name,
                "iou": None if np.isnan(iou_val) else float(iou_val),
                "intersection": int(inter_val),
                "union": int(union_val),
            }
            for name, iou_val, inter_val, union_val in zip(
                eval_names, selected_iou, selected_intersection, selected_union
            )
        ],
        "ignored_categories": [name for name in CATEGORY_NAMES if name not in eval_names],
    }

    output_json = args.output_json
    if output_json is None:
        output_json = Path("./outputs") / args.weather / "iou_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    import json

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved IoU summary to {output_json}")


if __name__ == "__main__":
    main()
