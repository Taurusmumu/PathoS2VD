# PathoS2VD

PathoS2VD generates a pseudo pathology z-stack from one 2D pathology image.
It learns an 11-plane source z-axis prior, adapts it to a target image domain,
and produces 33 planes at inference:

```text
z00–z10: upper extension
z11–z21: central stack
z22–z32: lower extension
```

## Install

Use Python 3.10 or 3.11:

```bash
pip install -r requirements.txt
```

Training and generation require access to the SVD-XT base model
`stabilityai/stable-video-diffusion-img2vid-xt` and user-provided data and
checkpoints. Configure paths with the `PATHOS2VD_*` environment variables used
in the YAML files; no machine-specific paths are stored in this repository.

## Data preparation

For raw source stacks arranged as:

```text
SOURCE_ROOT/<slide>/z00...z18/<patch>.png
```

build the training annotation with:

```bash
python scripts/build_blur_motion_annotation.py --source-root SOURCE_ROOT
```

This writes `SOURCE_ROOT/blur_motion_data6.csv`, which is read directly by the
source dataset loader.

## Training and generation

```bash
accelerate launch scripts/train_source_vae.py --config configs/stage1/vae.yaml
accelerate launch scripts/train_stage1.py --config configs/stage1/diffusion.yaml
accelerate launch scripts/train_target_vae.py --config configs/stage2/aggc.yaml
accelerate launch scripts/train_stage2.py --config configs/stage2/aggc.yaml

python scripts/generate.py --config configs/inference/default.yaml
```

See `configs/` for required environment variables and `docs/` for evaluation
and preprocessing details.

## Acknowledgements

- The defocus-blur implementation is adapted from
  [dfe-pr2010](https://github.com/alikaraali/dfe-pr2010).

- The Stable Video Diffusion training implementation is adapted from
  [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend).

- The VAE and diffusion components build on
  [Hugging Face Diffusers](https://github.com/huggingface/diffusers).