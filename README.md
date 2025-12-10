# ROB535 Final Project Semantic Segmentation

## YOLO-v8-seg
### Setup
1. Install YOLOv8-Seg by following the [official guide](https://github.com/ultralytics/ultralytics)
2. Verify installation by downloading a test image, and running `testYOLO.py` inside `./yolo_scripts`. You will need to modify the image path.
3. Download the RGB images and class-segmentation labels from the [VKITTI2 dataset](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/). 
4. Modify the ROOT path in `./yolo_scripts/new_eval.py` with you dataset path
### Evaluation and Visualization
Run the following command to get the IoUs and mIoU evaluation result, you need the semantic data.
```python
python new_eval.py
```

To get visualization, you may run the `testYOLO.py` file for single frame outputs.

## SAM3
...existing content or project details...

## CLIP+VGGT
### Setup
1. Install VGGT by following the [official guide](https://github.com/facebookresearch/vggt), but replace their default `requirements.txt` install step with our file `./vggt_scripts/requirements.txt`:
2. Download the RGB images and class-segmentation labels from the [VKITTI2 dataset](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/).
3. Arrange the downloaded RGB frames as `./vggt/test_data/scene_{xx}/{weather}` (for example, `scene_01/sunset`).
4. Arrange the corresponding semantic labels as `./vggt/gt_seg/scene_{xx}/{weather}`.
5. Copy `./vggt_scripts/my_demo_vggt.py` and `./vggt_scripts/evaluate.py` into the root `./vggt` directory so the demo and evaluation commands can find them.

### Demo Script
Run the following command to get visualization and predicted semantic segmentation.
```python
python my_demo_vggt.py
```

### Evaluation
Run the following command to get the IoUs and mIoU evaluation result, you need the semantic data.
```python
python evaluate.py --weather {WEATHER} --pred-path {OUTPUT_FOLDER}/{WEATHER}/semantic_map.npz
```
