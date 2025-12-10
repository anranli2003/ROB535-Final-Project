import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegModel, CLIPSegForImageSegmentation

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEGMENT_PROMPTS = [
    "terrain",
    "sky",
    "tree",
    "vegetation",
    "building",
    "road",
    "guardrail",
    "traffic sign",
    "traffic light",
    "pole",
    "misc",
    "truck",
    "car",
    "van",
]
SEGMENT_COLORS = {
    "terrain": (210, 0, 200),
    "sky": (90, 200, 255),
    "tree": (0, 199, 0),
    "vegetation": (90, 240, 0),
    "building": (140, 140, 140),
    "road": (100, 60, 100),
    "guardrail": (250, 100, 255),
    "traffic sign": (255, 255, 0),
    "traffic light": (200, 200, 0),
    "pole": (255, 130, 0),
    "misc": (80, 80, 80),
    "truck": (160, 60, 60),
    "car": (255, 127, 80),
    "van": (0, 139, 139),
}


BLEND_ALPHA = 0.58
REG_LAMBDA = 1e-3
WEATHER = "clone"


def load_vggt() -> VGGT:
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


def build_text_encoder() -> Tuple[CLIPSegProcessor, CLIPSegModel]:
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    text_model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    text_model.eval()
    return processor, text_model


def build_segmentation_model(processor: CLIPSegProcessor) -> CLIPSegForImageSegmentation:
    # CLIPSegProcessor already caches tokenizer/vision config; reuse for segmentation.
    segment_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    segment_model.eval()
    return segment_model


def compute_teacher_probabilities(
    rgb_np: np.ndarray,
    processor: CLIPSegProcessor,
    segment_model: CLIPSegForImageSegmentation,
    prompts: List[str],
) -> torch.Tensor:
    pil_image = Image.fromarray(rgb_np)
    batched_images = [pil_image] * len(prompts)
    inputs = processor(
        text=prompts,
        images=batched_images,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = segment_model(**inputs)
    probs = torch.sigmoid(outputs.logits)
    if probs.dim() == 4:
        probs = probs.squeeze(1)
    probs = torch.nn.functional.interpolate(
        probs.unsqueeze(0),
        size=rgb_np.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return probs


def gather_feature_tokens(vggt_model: VGGT, images: torch.Tensor) -> torch.Tensor:
    images_batched = images.unsqueeze(0) if images.dim() == 4 else images
    with torch.no_grad():
        token_list, patch_start_idx = vggt_model.aggregator(images_batched)
    fused_tokens = token_list[-1]
    return fused_tokens[:, :, patch_start_idx:, :]


def reshape_tokens(tokens: torch.Tensor, original_hw: Tuple[int, int], patch_size: int) -> torch.Tensor:
    B, S, num_tokens, dim = tokens.shape
    H, W = original_hw
    patch_h = H // patch_size
    patch_w = W // patch_size
    tokens = tokens.view(B, S, patch_h, patch_w, dim)
    return tokens.permute(0, 1, 4, 2, 3)


def get_text_embeddings(
    processor: CLIPSegProcessor,
    clipseg_model: CLIPSegModel,
    prompts: List[str],
) -> torch.Tensor:
    tokenized = processor.tokenizer(prompts, padding=True, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        sentence_embeddings = clipseg_model.get_text_features(**tokenized)
    return torch.nn.functional.normalize(sentence_embeddings, dim=1)


def compute_prompt_similarity(feature_map: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    flattened = feature_map.view(feature_map.shape[0], -1).permute(1, 0)
    flattened = torch.nn.functional.normalize(flattened, dim=1)
    sims = flattened @ text_embeddings.T
    sims = sims.permute(1, 0)
    return sims.view(text_embeddings.shape[0], feature_map.shape[1], feature_map.shape[2])


def generate_segmentations(
    features: torch.Tensor,
    text_embeddings: torch.Tensor,
    prompts: List[str],
    rgb_np: np.ndarray,
    upscale_size: Tuple[int, int],
) -> Tuple[np.ndarray, torch.Tensor]:
    feature_dim = text_embeddings.shape[1]
    if features.shape[0] < feature_dim:
        raise ValueError("VGGT features have lower dimensionality than text embeddings.")

    aligned = features[:feature_dim]
    similarity_maps = compute_prompt_similarity(aligned, text_embeddings)
    similarity_maps = torch.nn.functional.interpolate(
        similarity_maps.unsqueeze(0),
        size=upscale_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    prompt_mean = similarity_maps.mean(dim=(1, 2), keepdim=True)
    prompt_std = similarity_maps.std(dim=(1, 2), keepdim=True)
    normalized_maps = similarity_maps - prompt_mean
    normalized_maps = torch.where(prompt_std > 1e-6, normalized_maps / (prompt_std + 1e-6), normalized_maps)

    probabilities = torch.softmax(normalized_maps, dim=0)
    winner_idx = probabilities.argmax(dim=0)
    combined = rgb_np.copy().astype(np.float32) / 255.0
    for class_idx, prompt in enumerate(prompts):
        mask = winner_idx == class_idx
        if mask.any():
            color = np.array(SEGMENT_COLORS[prompt]) / 255.0
            mask_np = mask.cpu().numpy()
            combined[mask_np] = combined[mask_np] * (1.0 - BLEND_ALPHA) + color * BLEND_ALPHA

    return np.clip(combined, 0, 1), probabilities


def main():
    rgb_dir = Path(f"/home/yehengz/vggt/test_data/scene_01/{WEATHER}/frames/rgb/Camera_0")
    image_paths = sorted(str(p) for p in rgb_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {rgb_dir}")
    image_paths = image_paths[::8]

    images = load_and_preprocess_images(image_paths).to(device)
    vggt_model = load_vggt()
    processor, clipseg_text_model = build_text_encoder()
    segment_model = build_segmentation_model(processor)

    patch_tokens = gather_feature_tokens(vggt_model, images)
    feature_maps = reshape_tokens(patch_tokens, images.shape[-2:], vggt_model.aggregator.patch_size)
    text_embeddings = get_text_embeddings(processor, clipseg_text_model, SEGMENT_PROMPTS).to(device).float()
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)

    output_dir = Path(f"./outputs_vggt_features/{WEATHER}")
    output_dir.mkdir(parents=True, exist_ok=True)
    legend_handles = [mpatches.Patch(color=np.array(SEGMENT_COLORS[prompt]) / 255.0, label=prompt.title()) for prompt in SEGMENT_PROMPTS]
    semantic_maps: List[np.ndarray] = []

    num_prompts = len(SEGMENT_PROMPTS)
    embed_dim = feature_maps.shape[2]
    XtX = torch.zeros((embed_dim, embed_dim), device=device, dtype=torch.float32)
    XtY = torch.zeros((embed_dim, num_prompts), device=device, dtype=torch.float32)
    frame_cache: List[Dict[str, object]] = []

    print("Collecting VGGT features and CLIPSeg pseudo-labels for projection fit...")
    for frame_idx, (img_path, feature) in enumerate(zip(image_paths, feature_maps[0])):
        rgb = Image.open(img_path).convert("RGB")
        rgb_np = np.array(rgb)

        teacher_probs = compute_teacher_probabilities(rgb_np, processor, segment_model, SEGMENT_PROMPTS)
        teacher_probs = teacher_probs.to(device=device, dtype=torch.float32)
        feature = feature.to(device=device, dtype=torch.float32)

        teacher_patch = torch.nn.functional.interpolate(
            teacher_probs.unsqueeze(0),
            size=feature.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        feat_flat = feature.view(feature.shape[0], -1).permute(1, 0)
        target_flat = teacher_patch.view(num_prompts, -1).permute(1, 0)

        XtX += feat_flat.t() @ feat_flat
        XtY += feat_flat.t() @ target_flat

        frame_cache.append(
            {
                "feature": feature.detach().cpu().half(),
                "image_path": img_path,
                "rgb_shape": rgb_np.shape[:2],
            }
        )

    reg_eye = REG_LAMBDA * torch.eye(embed_dim, device=device, dtype=torch.float32)
    proj_weights = torch.linalg.solve(XtX + reg_eye, XtY)

    lstsq_solution = torch.linalg.lstsq(text_embeddings, proj_weights.T)
    feature_to_clip = lstsq_solution.solution.T.contiguous()

    print("Learned projection matrix computed in-memory.")

    for frame_idx, cache in enumerate(frame_cache):
        feature = cache["feature"].to(device=device, dtype=torch.float32)
        patch_h, patch_w = feature.shape[1], feature.shape[2]

        feat_flat = feature.view(feature.shape[0], -1).permute(1, 0)
        projected = feat_flat @ feature_to_clip
        projected = torch.nn.functional.normalize(projected, dim=1)
        logits = projected @ text_embeddings.T
        probabilities = torch.softmax(logits, dim=1)

        prob_maps = probabilities.permute(1, 0).view(num_prompts, patch_h, patch_w)
        prob_maps_up = torch.nn.functional.interpolate(
            prob_maps.unsqueeze(0),
            size=cache["rgb_shape"],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        winner_idx = prob_maps_up.argmax(dim=0)

        rgb = Image.open(cache["image_path"]).convert("RGB")
        rgb_np = np.array(rgb).astype(np.float32) / 255.0
        winner_np = winner_idx.cpu().numpy()

        label_map = np.zeros_like(winner_np, dtype=np.uint8)
        for class_idx, _prompt in enumerate(SEGMENT_PROMPTS, start=1):
            label_map[winner_np == (class_idx - 1)] = class_idx
        semantic_maps.append(label_map)

        combined = rgb_np.copy()
        for class_idx, prompt in enumerate(SEGMENT_PROMPTS):
            class_mask = winner_np == class_idx
            if not np.any(class_mask):
                continue
            color = np.array(SEGMENT_COLORS[prompt]) / 255.0
            combined[class_mask] = combined[class_mask] * (1.0 - BLEND_ALPHA) + color * BLEND_ALPHA

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(combined)
        ax.axis("off")
        ax.legend(handles=legend_handles, loc="lower right")
        combined_path = output_dir / f"feature_seg_frame_{frame_idx:02d}.png"
        fig.tight_layout()
        fig.savefig(combined_path, dpi=150)
        plt.close(fig)

        print(f"Saved feature-based segmentation to {combined_path}")

    if semantic_maps:
        semantic_array = np.stack(semantic_maps, axis=0)
        semantic_map_path = output_dir / "semantic_map.npy"
        np.save(semantic_map_path, semantic_array)
        semantic_map_npz_path = output_dir / "semantic_map.npz"
        np.savez_compressed(
            semantic_map_npz_path,
            semantic_map=semantic_array,
            prompts=np.array(SEGMENT_PROMPTS, dtype=object),
        )
        print(f"Saved stacked semantic maps to {semantic_map_path} and {semantic_map_npz_path}")
    else:
        print("No semantic maps were generated; skipping serialization.")


if __name__ == "__main__":
    main()