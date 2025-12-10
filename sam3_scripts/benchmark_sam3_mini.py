#!/usr/bin/env python3
"""
SAM3 Evaluation on VKITTI Dataset

Evaluates SAM3 segmentation performance on Virtual KITTI dataset.
Prompts SAM3 with COCO classes and calculates mIoU against ground truth masks.

Usage:
    python benchmark_sam3.py --scene Scene20 --weather morning --camera Camera_0
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import csv
import sys

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from coco_classes import COCO_CLASSES
from true_segmentation_labels import TRUE_SEGMENTATION_LABELS


# PROMPTED_CLASSES: What we ask SAM3 to search for
PROMPTED_CLASSES = ["car", "truck", "traffic light"]

# EVALUATED_CLASSES: Which VKITTI classes to include in mIoU
# Must be subset of TRUE_SEGMENTATION_LABELS.keys()
EVALUATED_CLASSES = ["car", "truck", "trafficlight"]

# PROMPT_TO_EVAL_MAPPING: Maps PROMPTED class -> EVALUATED class
PROMPT_TO_EVAL_MAPPING = {
    "car": "car",
    "truck": "truck",
    "traffic light": "trafficlight", 
}

# Reverse mapping for convenience
EVAL_TO_PROMPT_MAPPING = {v: k for k, v in PROMPT_TO_EVAL_MAPPING.items()}



def parse_gt_mask(gt_path):

    gt_img = np.array(Image.open(gt_path))  # (H, W, 3)
    masks = {}
    for class_name, rgb in TRUE_SEGMENTATION_LABELS.items():
        if class_name == "undefined":
            continue  # Skip undefined class
        mask = np.all(gt_img == rgb, axis=-1)
        masks[class_name] = mask
    return masks


def run_sam3_inference(processor, image_path):

    image = Image.open(image_path)
    W, H = image.size  # PIL gives (width, height)

    # Initialize empty masks for ALL evaluated classes
    pred_masks = {c: np.zeros((H, W), dtype=bool) for c in EVALUATED_CLASSES}

    inference_state = processor.set_image(image)

    for prompt_class in PROMPTED_CLASSES:
        processor.reset_all_prompts(inference_state)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_class)

        masks = output["masks"]  # (N, H, W) boolean tensor

        if prompt_class in PROMPT_TO_EVAL_MAPPING and len(masks) > 0:
            eval_class = PROMPT_TO_EVAL_MAPPING[prompt_class]
            if eval_class in EVALUATED_CLASSES:
                combined_mask = masks.any(dim=0).cpu().numpy()
                pred_masks[eval_class] = np.logical_or(
                    pred_masks[eval_class],
                    combined_mask
                )

    return pred_masks


def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU between two binary masks.

    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)

    Returns:
        float or None: IoU score in [0, 1], or None if GT is empty (class not present)
    """
    gt_pixels = gt_mask.sum()

    if gt_pixels == 0:
        return None

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection) / float(union)


def validate_gt_colors(gt_dir, num_samples=5):
    """Validate that GT mask colors match TRUE_SEGMENTATION_LABELS.

    Args:
        gt_dir: Path to GT mask directory
        num_samples: Number of random masks to check

    Returns:
        tuple: (is_valid, found_colors, unknown_colors)
    """
    gt_files = sorted(gt_dir.glob("classgt_*.png"))[:num_samples]

    known_colors = {tuple(rgb): name for name, rgb in TRUE_SEGMENTATION_LABELS.items()}
    found_colors = set()
    unknown_colors = set()

    for gt_path in gt_files:
        img = np.array(Image.open(gt_path))
        pixels = img.reshape(-1, img.shape[-1])
        unique_colors = np.unique(pixels, axis=0)

        for color in unique_colors:
            color_tuple = tuple(color)
            found_colors.add(color_tuple)
            if color_tuple not in known_colors:
                unknown_colors.add(color_tuple)

    return len(unknown_colors) == 0, found_colors, unknown_colors


def evaluate(args):
    """Main evaluation loop.

    Args:
        args: Parsed command line arguments

    Returns:
        tuple: (per_class_results dict, overall_miou float)
    """
    # Build paths
    rgb_base = Path("/home/lujust/src/maani-sam3/vkitti_2.0.3_rgb")
    gt_base = Path("/home/lujust/src/maani-sam3/vkitti_2.0.3_classSegmentation")

    # Get image list
    rgb_dir = rgb_base / args.scene / args.weather / "frames/rgb" / args.camera
    gt_dir = gt_base / args.scene / args.weather / "frames/classSegmentation" / args.camera

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    # Validate GT colors before running
    print("Validating GT mask colors...")
    is_valid, found_colors, unknown_colors = validate_gt_colors(gt_dir)
    if not is_valid:
        print(f"WARNING: Found {len(unknown_colors)} unknown colors in GT masks!")
        for color in unknown_colors:
            print(f"  Unknown: RGB{color}")
        print("These pixels will be ignored. Check TRUE_SEGMENTATION_LABELS!")
    else:
        print(f"GT colors validated OK ({len(found_colors)} unique colors found)")

    image_files = sorted(rgb_dir.glob("rgb_*.jpg"))
    print(f"Found {len(image_files)} images to process")

    # Load SAM3 model
    print("Loading SAM3 model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)
    print("Model loaded!")

    # Accumulate IoU per class (ONLY for EVALUATED_CLASSES)
    class_ious = {c: [] for c in EVALUATED_CLASSES}

    for img_path in tqdm(image_files, desc="Processing images"):
        frame_id = img_path.stem.replace("rgb_", "")
        gt_path = gt_dir / f"classgt_{frame_id}.png"

        if not gt_path.exists():
            print(f"Warning: GT not found for {img_path.name}, skipping")
            continue

        # Get predictions and ground truth
        pred_masks = run_sam3_inference(processor, img_path)
        gt_masks = parse_gt_mask(gt_path)

        # Calculate IoU for each EVALUATED class only
        for class_name in EVALUATED_CLASSES:
            pred = pred_masks.get(class_name)
            gt = gt_masks.get(class_name)

            if pred is None or gt is None:
                continue

            iou = calculate_iou(pred, gt)
            # iou is None if GT has no pixels for this class (skip)
            if iou is not None:
                class_ious[class_name].append(iou)

    # Compute mean IoU per class (only for frames where class appeared)
    results = {}
    class_counts = {}  # Track how many frames each class appeared in
    for class_name, ious in class_ious.items():
        class_counts[class_name] = len(ious)
        results[class_name] = np.mean(ious) if ious else None  # None if class never appeared

    # Overall mIoU (only over classes that appeared at least once)
    valid_ious = [iou for iou in results.values() if iou is not None]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    num_classes_evaluated = len(valid_ious)

    return results, miou, class_counts, num_classes_evaluated


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 on VKITTI dataset")
    parser.add_argument("--scene", default="Scene20",
                        help="Scene to evaluate (Scene01, Scene02, Scene06, Scene18, Scene20)")
    parser.add_argument("--weather", default="morning",
                        help="Weather condition (clone, fog, morning, overcast, rain, sunset)")
    parser.add_argument("--camera", default="Camera_0",
                        help="Camera to use (Camera_0 or Camera_1)")
    parser.add_argument("--output", default="sam3_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("SAM3 Evaluation Configuration")
    print("=" * 50)
    print(f"PROMPTED_CLASSES: {len(PROMPTED_CLASSES)} classes")
    print(f"EVALUATED_CLASSES: {len(EVALUATED_CLASSES)} classes")
    print(f"Mappings: {PROMPT_TO_EVAL_MAPPING}")
    print(f"Scene: {args.scene}/{args.weather}/{args.camera}")
    print("=" * 50)

    # Run evaluation
    results, miou, class_counts, num_classes_evaluated = evaluate(args)

    # Print results
    print("\n=== Per-Class IoU ===")
    for class_name, iou in results.items():
        has_mapping = class_name in EVAL_TO_PROMPT_MAPPING
        match_status = "MAPPED" if has_mapping else "NO MAP"
        marker = "+" if has_mapping else "-"
        count = class_counts[class_name]

        if iou is not None:
            print(f"  [{marker}] {match_status:8} {class_name:15} IoU: {iou:.4f}  (in {count} frames)")
        else:
            print(f"  [{marker}] {match_status:8} {class_name:15} IoU: N/A     (not in scene)")

    print(f"\n=== mIoU: {miou:.4f} ===")
    print(f"    (averaged over {num_classes_evaluated}/{len(EVALUATED_CLASSES)} classes that appeared in scene)")

    # Save to CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou", "frames_appeared", "has_prompt_mapping", "mapped_to"])
        for class_name, iou in results.items():
            mapped_to = EVAL_TO_PROMPT_MAPPING.get(class_name, "")
            iou_str = f"{iou:.6f}" if iou is not None else "N/A"
            writer.writerow([class_name, iou_str, class_counts[class_name],
                           class_name in EVAL_TO_PROMPT_MAPPING, mapped_to])
        writer.writerow(["mIoU", f"{miou:.6f}", num_classes_evaluated, "", ""])

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
