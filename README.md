# OpenVINO Generative AI Scripts

This repository contains Python scripts for **running and evaluating OpenVINO-based generative AI models**.  
They are adapted from official OpenVINO Jupyter notebooks and customized for **our research objectives**, including large-scale image generation and model quality benchmarking.

---

## 📌 Overview

- **Purpose**: Automate inference, image generation, and evaluation for multiple generative AI models using OpenVINO Runtime.
- **Dataset**: [`phiyodr/coco2017`](https://huggingface.co/datasets/phiyodr/coco2017) (COCO 2017 prompts for text-to-image generation)
- **Core Features**:
  - Support for multiple models and precision formats (weights: **int4**, **int8**, **fp16**)
  - Automated folder creation for generated outputs
  - Large-batch inference: **300 images** per run with prompts from COCO 2017
  - Pre-written evaluation scripts for **IS**, **CLIP Score**, **FID**, **PickScore**

---

## 🛠 Supported Models

| Model Name | Variants (Weights) | Script Example |
|------------|--------------------|----------------|
| **FLUX.1 Schnell** | int4 / int8 / fp16 | `flux_int8.py` |
| **Stable Diffusion XL** | int4 / int8 / fp16 | `sdxl_fp16.py` |
| **Stable Diffusion v1.5** | int4 / int8 / fp16 | `stable-diffusion-v1.5-int4.py` |

### ⚠️ Special Note on SDXL Models

The official SDXL repository does **not** provide different weight formats by default.  
You will need to manually download and export them with `optimum-cli` before running the scripts.

**Example: Download SDXL int4 model**
```bash
optimum-cli export openvino \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --weight-format int4 \
    --dataset conceptual_captions \
    int4_sdxl/
```
After downloading, update the model_path in the corresponding Python script to point to your local folder, e.g.:
```bash
model_path = "int4_sdxl"
```
## 📊 Workflow
1. **Run Model Script**
   - Each script will:
     - Load the selected model in OpenVINO Runtime
     - Use prompts from `phiyodr/coco2017`
     - Generate **300 images** per run
     - Save results in a timestamped folder under `{model_name}/{weight}/images`

2. **Evaluate Generated Images**
   - Use the scripts in `evaluation/` to compute:
     - **IS** (Inception Score)
     - **CLIP Score**
     - **FID** (Fréchet Inception Distance)
     - **PickScore**

