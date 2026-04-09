# SDG-DSB: Spectral Degradation Guided Diffusion Schrödinger Bridge for Hyperspectral Images Super-Resolution

This is the official pytorch code for "**SDG-DSB: Spectral Degradation Guided Diffusion Schrödinger Bridge for Hyperspectral Images Super-Resolution**", which has been accepted by TGRS2025.

## Getting started

### Installation

```shell
conda env create --file requirements.yaml python=3
conda activate sdgdsb
```

### pre-trained checkpoints

Download the weight file `256x256_diffusion_uncond.pt` from [Google Drive](https://drive.google.com/drive/folders/1uop1IeM3DA2Z4vWfWBQpaTs6o0KpJSJK?usp=sharing) to the `data` directory, and download `latest.pt` to the `results/sr4x-bicubic` directory.

### Running

```shell
python sample.py --clip-denoise --use-cddb-deep
```

## Acknowledge

Some of the codes and pre-trained checkpoints are built upon [CDDB](https://github.com/hyungjin-chung/CDDB) and [I2SB](https://github.com/NVlabs/I2SB).