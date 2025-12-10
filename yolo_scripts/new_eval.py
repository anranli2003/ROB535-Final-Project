import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

ROOT = "/path/to/vkitti"
RGB_ROOT = f"{ROOT}/vkitti_rgb"
GT_ROOT  = f"{ROOT}/vkitti_classSeg"

SCENARIOS = ["clone", "morning", "rain", "sunset", "fog", "overcast"]
MODEL_PATH = f"{ROOT}/yolov8n-seg.pt"


GT_COLORS = {
    "car":           (80, 127, 255),   # (B, G, R)
    "truck":         (60, 60, 160),
    "traffic_light": (0, 200, 200),
}


YOLO_CLASS_IDS = {
    "car":           2,
    "truck":         7,
    "traffic_light": 9,
}

model = YOLO(MODEL_PATH)

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None
    return intersection / union


# MAIN LOOP
final_results = {}
for scenario in SCENARIOS:

    rgb_dir = f"{RGB_ROOT}/{scenario}/frames/rgb/Camera_0"
    gt_dir  = f"{GT_ROOT}/{scenario}/frames/classSegmentation/Camera_0"

    image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    print(f"Found {len(image_files)} images")

    per_class_ious = {cls: [] for cls in GT_COLORS.keys()}

    for img_name in tqdm(image_files):
        img_path = os.path.join(rgb_dir, img_name)
        frame_id = img_name.replace("rgb_", "").replace(".jpg", "")
        gt_path = os.path.join(gt_dir, f"classgt_{frame_id}.png")

        if not os.path.exists(gt_path):
            continue

        # LOAD GT IN COLOR 
        gt_rgb = cv2.imread(gt_path, cv2.IMREAD_COLOR)

        H, W, _ = gt_rgb.shape

        # Build GT masks per class via RGB comparison
        gt_masks = {}
        for cls_name, bgr in GT_COLORS.items():
            gt_masks[cls_name] = np.all(gt_rgb == np.array(bgr), axis=-1)

        # Run YOLO
        res = model(img_path, conf=0.25, device="cuda")[0]

        pred_masks = {cls: np.zeros((H, W), dtype=bool)
                      for cls in GT_COLORS.keys()}

        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()

            for i, cls_id in enumerate(classes):
                cls_id = int(cls_id)
                for cls_name, yolo_id in YOLO_CLASS_IDS.items():
                    if cls_id == yolo_id:
                        m = masks[i] > 0.5
                        m = cv2.resize(m.astype(np.uint8), (W, H)) > 0
                        pred_masks[cls_name] = np.logical_or(pred_masks[cls_name], m)

        # Compute IoU per class
        for cls_name in GT_COLORS.keys():
            iou = compute_iou(pred_masks[cls_name], gt_masks[cls_name])
            if iou is not None:
                per_class_ious[cls_name].append(iou)

    # Aggregate scenario results
    scenario_result = {}
    valid_vals = []

    for cls_name, vals in per_class_ious.items():
        if vals:
            mean_iou = float(np.mean(vals))
            scenario_result[cls_name] = mean_iou
            valid_vals.append(mean_iou)
            print(f"{scenario} {cls_name} IoU = {mean_iou:.4f}")
        else:
            scenario_result[cls_name] = None
            print(f"{scenario} {cls_name}: no valid IoU values")

    scenario_result["mIoU"] = float(np.mean(valid_vals)) if valid_vals else None
    final_results[scenario] = scenario_result

    print(f"{scenario} mIoU over classes = {scenario_result['mIoU']}")


print("\n========== FINAL PER-CLASS IoU RESULTS ==========")
for scenario, metrics in final_results.items():
    print(f"\n{scenario.upper()}")
    for cls_name in GT_COLORS.keys():
        print(f"{cls_name:15s}: {metrics[cls_name]}")
    print(f"mIoU           : {metrics['mIoU']}")

