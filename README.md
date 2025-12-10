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
### Setup
1. Install SAM3 by following the [official SAM 3 guide](https://github.com/facebookresearch/sam3). You will need to authenticate with Hugging Face (`huggingface-cli login`).
2. Download the RGB images and class-segmentation labels from the [VKITTI2 dataset](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/).
3. Update the `rgb_base` and `gt_base` paths in the benchmark scripts to point to your dataset location.

### Evaluation
We provide several benchmark scripts in `./sam3_scripts/` that evaluate SAM3's text-prompted segmentation on VKITTI:
```bash
# Scenes: 'Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20'
# Weather: 'clone', 'morning', 'rain', 'sunset', 'fog', 'overcast
# Cameras: 'Camera_0', 'Camera_1'
python benchmark_sam3.py --scene Scene20 --weather morning --camera Camera_0
```
This will output per-class IoU scores and mIoU to a CSV file. You can also try `benchmark_sam3.py` (prompts with COCO classes) or `benchmark_sam3_mini.py` (car/truck/traffic light only) depending on your use case.

NOTE: 
- The scripts ```benchmark_sam3_mini.py``` and ```benchmark_sam3_vkitti.py``` are essentially direct copies of ```benchmark_sam3.py```.
- All that was changed were the classes to be queued and evaluated on, which were purely out of convenience; for proper usage, edit `benchmark_sam3.py` to fit your needs.

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
