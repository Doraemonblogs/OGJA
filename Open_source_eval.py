#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from typing import List, Tuple, Optional

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from modelscope import StableDiffusionInpaintPipeline, AutoPipelineForInpainting

# labels for MMA Dataset 
labels = [0,0,0,0,1,0,1,1,0,0,0,0,0,0,3,0,0,2,0,0,0,1,2,0,3,2,1,1,0,4,1,1,0,0,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0]

def list_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str, str]]:
    """
    Match:
      image: i.png
      mask : i_mask.png
    Return: [(i_str, image_path, mask_path), ...] sorted by i (numeric).
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    items = []
    for name in os.listdir(image_dir):
        if not name.lower().endswith(".png"):
            continue
        stem = name[:-4]  # remove .png
        if not stem.isdigit():
            continue

        i_str = stem
        img_path = os.path.join(image_dir, f"{i_str}.png")
        msk_path = os.path.join(mask_dir, f"{i_str}_mask.png")
        if os.path.isfile(img_path) and os.path.isfile(msk_path):
            items.append((i_str, img_path, msk_path))

    items.sort(key=lambda x: int(x[0]))
    return items


def ensure_mask_is_grayscale(mask: Image.Image) -> Image.Image:
    """
    Inpaint mask convention:
      - white: inpaint
      - black: keep
    Ensure mask is single-channel (L).
    """
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask


def maybe_resize(img: Image.Image, size: Optional[int]) -> Image.Image:
    if size is None:
        return img
    if img.size == (size, size):
        return img
    return img.resize((size, size), resample=Image.BICUBIC)


def add_gaussian_noise_pil(img: Image.Image, std: float, generator: Optional[torch.Generator] = None) -> Image.Image:
    """
    Add Gaussian noise to a PIL RGB image in [0,1] space, then clamp and return PIL.
    std: noise standard deviation in [0,1] space (e.g., 0.05).
    """
    if std <= 0:
        return img
    if img.mode != "RGB":
        img = img.convert("RGB")

    # PIL -> torch [0,1], shape [3,H,W]
    x = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
    x = x.permute(2, 0, 1)

    noise = torch.randn(x.shape, generator=generator) * std
    x = (x + noise).clamp(0.0, 1.0)

    # torch -> PIL
    x = (x.permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(x, mode="RGB")


def load_pipeline(args, torch_dtype):
    """
    Choose correct pipeline:
    - SDXL inpainting: AutoPipelineForInpainting
    - SD1/SD2 inpainting: StableDiffusionInpaintPipeline
    """
    pipe_mode = args.pipeline.lower()

    if pipe_mode not in ["auto", "sd", "sdxl"]:
        raise ValueError("--pipeline must be one of: auto | sd | sdxl")

    model_l = args.model.lower()
    looks_like_sdxl = ("sdxl" in model_l) or ("stable-diffusion-xl" in model_l) or ("xl" in model_l)

    use_sdxl = (pipe_mode == "sdxl") or (pipe_mode == "auto" and looks_like_sdxl)

    if use_sdxl:
        pipe = AutoPipelineForInpainting.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            variant=args.variant if (args.variant and args.variant.lower() != "none") else None,
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            variant=args.variant if (args.variant and args.variant.lower() != "none") else None,
        )

    # Safety checker: some pipelines may not have this attribute, so handle compatibly
    if getattr(args, "disable_safety_checker", False):
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

    return pipe


def main():
    parser = argparse.ArgumentParser(description="Batch inpainting (ModelScope, SD/SDXL compatible).")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of original images, containing i.png")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of masks, containing i_mask.png")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory, saves i.png")
    parser.add_argument("--prompt", type=str, default="", help="Default positive prompt (used when prompt_list is empty)")
    parser.add_argument("--prompt_list", type=str, default="", help="Optional: prompt_list file path (json / txt). If provided, select prompt by labels[i]")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--gn", action="store_true", help="Whether to add random Gaussian noise to the input image before diffusion")
    parser.add_argument("--gn_std", type=float, default=0.05, help="Gaussian noise std (default 0.05, input normalized to [0,1])")
    parser.add_argument("--model", type=str, required=True, help="ModelScope/HF local path or model name")
    parser.add_argument("--pipeline", type=str, default="auto", choices=["auto", "sd", "sdxl"], help="auto: auto-detect; sd: force SD1/2 inpaint; sdxl: force SDXL inpaint")
    parser.add_argument("--variant", type=str, default="fp16", help="e.g. fp16 / None")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, -1 means random per image")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--strength", type=float, default=0.8, help="Inpaint strength (0~1, SDXL recommended <1.0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (watch VRAM)")
    parser.add_argument("--resize", type=int, default=1024, help="If set, resize image/mask to size×size. SDXL usually uses 1024.")
    parser.add_argument("--disable_safety_checker", action="store_true", default=True, help="Disable safety checker (research/audit only, do not use in public services)")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite output if it already exists (default: do not overwrite)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, please set --device cpu or install CUDA-enabled torch.")
    
    if not args.prompt_list and not args.prompt:
        raise ValueError("Either --prompt_list or --prompt must be provided.")

    # dtype
    torch_dtype = torch.float16 if (args.device == "cuda" and str(args.variant).lower() == "fp16") else torch.float32

    # --- load correct pipeline ---
    pipe = load_pipeline(args, torch_dtype=torch_dtype).to(args.device)

    # Optional: reduce VRAM usage
    if args.device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    prompt_list = None
    if args.prompt_list:
        if not os.path.isfile(args.prompt_list):
            raise FileNotFoundError(f"prompt_list file not found: {args.prompt_list}")

        with open(args.prompt_list, "r", encoding="utf-8") as f:
            prompt_list = json.load(f)

        if not isinstance(prompt_list, list) or len(prompt_list) == 0:
            raise ValueError("prompt_list must be a non-empty list")

        print(f"[INFO] Loaded prompt_list with {len(prompt_list)} prompts")
    else:
        print("[INFO] prompt_list not provided, using default prompt")

    pairs = list_pairs(args.image_dir, args.mask_dir)
    if not pairs:
        raise RuntimeError(
            f"No matched pairs found.\n"
            f"Expected images: {args.image_dir}/i.png and masks: {args.mask_dir}/i_mask.png (i is integer)."
        )

    def make_generator(seed: int):
        if args.device == "cuda":
            return torch.Generator(device="cuda").manual_seed(seed)
        return torch.Generator().manual_seed(seed)

    for start in tqdm(range(0, len(pairs), args.batch_size), desc="Inpainting"):
        chunk = pairs[start:start + args.batch_size]

        out_paths = [os.path.join(args.out_dir, f"{i_str}.png") for (i_str, _, _) in chunk]
        if (not args.overwrite) and all(os.path.exists(p) for p in out_paths):
            continue

        images: List[Image.Image] = []
        masks: List[Image.Image] = []
        i_list: List[str] = []

        for (i_str, img_path, msk_path) in chunk:
            img = Image.open(img_path).convert("RGB")
            msk = Image.open(msk_path)
            msk = ensure_mask_is_grayscale(msk)

            # SDXL commonly uses 1024; if your data is not 1024, it is recommended to use --resize 1024
            img = maybe_resize(img, args.resize)
            msk = maybe_resize(msk, args.resize)

            # Optional: add Gaussian noise to the input image BEFORE feeding into the diffusion model
            if args.gn:
                # For reproducibility: if seed >= 0, each image gets a deterministic noise seed
                if args.seed is None or args.seed < 0:
                    gn_gen = None
                else:
                    gn_seed = args.seed + int(i_str)  # per-image deterministic
                    gn_gen = torch.Generator().manual_seed(gn_seed)
                img = add_gaussian_noise_pil(img, std=args.gn_std, generator=gn_gen)

            images.append(img)
            masks.append(msk)
            i_list.append(i_str)

        if args.seed is None or args.seed < 0:
            generator = None
        else:
            generator = make_generator(args.seed + start)
        
        batch_prompts: List[str] = []
        for i_str in i_list:
            if prompt_list is not None:
                idx0 = int(i_str) - 1  # KEY: filenames start from 1
                if idx0 < 0 or idx0 >= len(labels):
                    raise IndexError(f"File {i_str}.png -> idx0={idx0} out of labels range (len={len(labels)})")

                label_idx = labels[idx0]
                if label_idx < 0 or label_idx >= len(prompt_list):
                    raise IndexError(f"labels[{idx0}]={label_idx} out of prompt_list range (len={len(prompt_list)})")

                p = prompt_list[label_idx]
            else:
                p = args.prompt

            batch_prompts.append(p)

        # Batch-compatible call: prompt/list + image/list + mask/list
        result = pipe(
            prompt=batch_prompts,
            negative_prompt=[args.negative_prompt] * len(images) if args.negative_prompt else None,
            image=images,
            mask_image=masks,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            generator=generator,
        ).images

        for i_str, out_path, out_img in zip(i_list, out_paths, result):
            if (not args.overwrite) and os.path.exists(out_path):
                continue
            out_img.save(out_path)

    print(f"Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
