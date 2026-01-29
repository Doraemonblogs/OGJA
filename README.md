# OGJA: One-Glance Jailbreak Attack

A toolkit for studying **image-only jailbreak attacks** against Commercial diffusion models. 
![intro](src\pipeline.png)

This repository provides:

* **Closed-source attack**: an FGSM-style perturbation generator guided by an ensemble of CLIP vision encoders (ViT-B/16, ViT-B/32, ViT-L/336, LAION), with optional local cropping for robust feature alignment.
* **Open-source evaluation**: a Stable Diffusion / SDXL inpainting runner for batch testing with prompts and masks.
* **Safety metric**: a NudeNet-based ASR (Attack Success Rate) checker to quantify how often generated images bypass nudity filters.

> ⚠️ **Disclaimer**: This code is intended **solely for academic research and safety analysis**. Do not use it for illegal, unethical, or non-consensual purposes.

![intro](src\intro.png)

## Project layout

```text
.
├── Closed_source_attack.py    # Hydra-driven FGSM attack (OGJA core)
├── Open_source_eval.py        # Open-source diffusion inpainting evaluation
├── config/
│   └── OGJA.yaml              # Default experiment configuration
├── surrogates/
│   └── FeatureExtractors/     # CLIP-based encoders and ensemble loss
├── utils.py                   # Shared helpers (cropping, logging, ASR, diffusion loaders)
├── src/
│   ├── intro.png              # Method overview figure
│   └── pipeline.png           # OGJA pipeline diagram
└── README.md
```

---

## Environment

* **Python**: 3.10+ (recommended)
* **Hardware**: CUDA-capable GPU strongly recommended

Install PyTorch matching your CUDA version, then install dependencies:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install hydra-core omegaconf tqdm pillow pandas einops wandb \
            transformers diffusers[torch] modelscope nudenet
```

Models (CLIP variants, Stable Diffusion / SDXL) are downloaded automatically on first run.
Set `HF_HOME` or `MODELSCOPE_CACHE` if you need custom cache locations.

---

## Data preparation

### Closed-source attack

The attack expects **two ImageFolder-style datasets with matching order**:

* `data.cle_data_path`: clean / source images
* `data.tgt_data_path`: target (reference) images

Directory structure example:

```text
dataset/
├── clean_images/
│   └── class_name/img001.jpg
└── target_images/
    └── class_name/img001.jpg
```

Images are paired **by index** after loading; ensure both folders contain the same number of images in the same order.

Default paths are defined in `config/OGJA.yaml` and can be overridden via the CLI.

---

## Run the closed-source attack

```bash
python Closed_source_attack.py \
  data.cle_data_path=/path/to/clean_images \
  data.tgt_data_path=/path/to/target_images \
  data.output=./our_results \
  optim.alpha=1.0 optim.epsilon=16 optim.steps=300 \
  model.backbone=[B16,B32,Laion] model.input_res=512 \
  model.device=cuda
```

### Output

Results are saved to:

```text
our_results/img/<config_hash>/<class_name>/image.png
```

The `config_hash` is automatically generated from the experiment configuration to keep runs isolated and reproducible.

### Key options

* **Cropping augmentation**

  * `model.use_source_crop` / `model.use_target_crop`
  * Window size, stride, and update interval are configurable

* **Ensemble vs single encoder**

  * Set `model.ensemble=false` and choose a single backbone if desired

* **Logging**

  * Enable Weights & Biases with:

    ```bash
    wandb.enabled=true wandb.project=YOUR_PROJECT wandb.entity=YOUR_ENTITY
    ```

---

## Run the open-source inpainting evaluation

```bash
python Open_source_eval.py \
  --image_dir data/images \
  --mask_dir data/masks \
  --out_dir outputs/inpaint \
  --prompt_list prompts.json \  # or --prompt "a default prompt"
  --model your/sd-or-sdxl-inpaint \
  --pipeline auto \              # auto | sd | sdxl
  --resize 1024 \
  --device cuda \
  --disable_safety_checker
```

Notes:

* Images are processed in batches; optional Gaussian noise can be enabled with `--gn --gn_std 0.05`.
* Filenames must follow `i.png` with corresponding masks `i_mask.png`.
* `prompts.json` is indexed by the `labels` list defined in the script.

---

## Configuration quick reference (`config/OGJA.yaml`)

* **data**: `batch_size`, `num_samples`, `output`, `cle_data_path`, `tgt_data_path`
* **optim**: `alpha`, `epsilon` (L∞ bound, 0–255 scale), `steps`
* **model**: `input_res`, `use_source_crop`, `use_target_crop`, `ensemble`, `device`, `backbone`,
  `window_ratio` / `area_scale_range`, `stride`, `interval`
* **wandb**: `enabled`, `project`, `entity`

---

## Practical tips

* **GPU memory**: reduce `batch_size` or input resolution if you encounter OOM errors.
* **Reproducibility**: random seeds are fixed in `Closed_source_attack.py` via `set_environment()`.
* **Safety**: `Open_source_eval.py` disables the diffusion safety checker by default; re-enable it outside controlled research settings.


