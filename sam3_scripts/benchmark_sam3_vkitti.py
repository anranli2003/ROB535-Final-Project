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
PROMPTED_CLASSES = [
    "terrain", "sky", "tree", "vegetation", "building", "road",
    "guardrail", "trafficsign", "trafficlight", "pole", "misc",
    "truck", "car", "van"
]

# EVALUATED_CLASSES: Which VKITTI classes to include in mIoU
# Must be subset of TRUE_SEGMENTATION_LABELS.keys()
EVALUATED_CLASSES = [
    "terrain", "sky", "tree", "vegetation", "building", "road",
    "guardrail", "trafficsign", "trafficlight", "pole", "misc",
    "truck", "car", "van"
]

# EVALUATED_CLASSES = ["car", "truck", "trafficlight"]

# PROMPT_TO_EVAL_MAPPING: Maps PROMPTED class -> EVALUATED class
PROMPT_TO_EVAL_MAPPING = {
    "terrain": "terrain",
    "sky": "sky",
    "tree": "tree",
    "vegetation": "vegetation",
    "building": "building",
    "road": "road",
    "guardrail": "guardrail",
    "trafficsign": "trafficsign",
    "trafficlight": "trafficlight",
    "pole": "pole",
    "misc": "misc",
    "truck": "truck",
    "car": "car",
    "van": "van",
}

EVAL_TO_PROMPT_MAPPING = {v: k for k, v in PROMPT_TO_EVAL_MAPPING.items()}



def parse_gt_mask(gt_path):

    gt_img = np.array(Image.open(gt_path))  # (H, W, 3)
    masks = {}
    for class_name, rgb in TRUE_SEGMENTATION_LABELS.items():
        if class_name == "undefined":
            continue  # Skip undefined class
        # Create binary mask where pixels match the class color
        mask = np.all(gt_img == rgb, axis=-1)  # Boolean mask (H, W)
        masks[class_name] = mask
    return masks


def run_sam3_inference(processor, image_path):

    image = Image.open(image_path)
    W, H = image.size  # PIL gives (width, height)

    # Initialize empty masks for ALL evaluated classes
    pred_masks = {c: np.zeros((H, W), dtype=bool) for c in EVALUATED_CLASSES}

    # Set image ONCE (expensive - runs vision backbone)
    inference_state = processor.set_image(image)

    # Prompt with ALL PROMPTED_CLASSES (cheap - reuses cached image features)
    for prompt_class in PROMPTED_CLASSES:
        processor.reset_all_prompts(inference_state)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt_class)

        masks = output["masks"]  # (N, H, W) boolean tensor

        # Only keep if this prompted class maps to an evaluated class
        if prompt_class in PROMPT_TO_EVAL_MAPPING and len(masks) > 0:
            eval_class = PROMPT_TO_EVAL_MAPPING[prompt_class]
            if eval_class in EVALUATED_CLASSES:
                # Union all instance masks into one semantic mask
                combined_mask = masks.any(dim=0).cpu().numpy()
                pred_masks[eval_class] = np.logical_or(
                    pred_masks[eval_class],
                    combined_mask
                )

    return pred_masks


def calculate_iou(pred_mask, gt_mask):

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0  # No GT and no prediction = 0 IoU
    return float(intersection) / float(union)


def evaluate(args):

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
            class_ious[class_name].append(iou)

    # Compute mean IoU per class
    results = {}
    for class_name, ious in class_ious.items():
        results[class_name] = np.mean(ious) if ious else 0.0

    # Overall mIoU (average over EVALUATED_CLASSES only)
    miou = np.mean(list(results.values()))

    return results, miou


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
    results, miou = evaluate(args)

    # Print results
    print("\n=== Per-Class IoU ===")
    for class_name, iou in results.items():
        has_mapping = class_name in EVAL_TO_PROMPT_MAPPING
        match_status = "MAPPED" if has_mapping else "NO MAP"
        marker = "+" if has_mapping else "-"
        print(f"  [{marker}] {match_status:8} {class_name:15} IoU: {iou:.4f}")

    print(f"\n=== mIoU: {miou:.4f} ===")
    print(f"    (averaged over {len(EVALUATED_CLASSES)} evaluated classes)")

    # Save to CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou", "has_prompt_mapping", "mapped_to"])
        for class_name, iou in results.items():
            mapped_to = EVAL_TO_PROMPT_MAPPING.get(class_name, "")
            writer.writerow([class_name, iou, class_name in EVAL_TO_PROMPT_MAPPING, mapped_to])
        writer.writerow(["mIoU", miou, "", ""])

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
