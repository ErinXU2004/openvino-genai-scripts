# 🧠 openvino-genai-scripts

> Python scripts for **text-to-image generation** using OpenVINO-accelerated AI PC models from Hugging Face, with quantitative visual comparison across model types and precision formats.

---

## 📌 Project Overview

This project provides a collection of Python scripts that run text-to-image generation using **OpenVINO** and state-of-the-art generative models. It is designed to:

- Benchmark **different models** (e.g., SDXL, Stable Diffusion v1.5, FLUX.1)
- Compare **different weight precisions** (INT4, INT8, FP16)
- Use **realistic text prompts** from [COCO2017 captions dataset](https://hf-mirror.com/datasets/phiyodr/coco2017)
- Store generated images and logs for further visual and quantitative evaluation

---

## 🧰 Supported Models & Weights

| Model Name | Precision Support       | Notes                                |
|------------|-------------------------|--------------------------------------|
| `SDXL`     | ✅ INT4 / ✅ INT8 / ✅ FP16 | Requires `openvino-sdxl` IR model     |
| `SD v1.5`  | ✅ INT4 / ✅ INT8 / ✅ FP16 | Lightweight & efficient baseline     |
| `FLUX.1`   | ✅ INT4 / ✅ INT8 / ✅ FP16 | Compact model for AI PC deployment   |

All models are sourced from Hugging Face and converted to **OpenVINO IR** format.

---

## 🧪 Dataset

The evaluation is based on text captions from:

> 📂 [`phiyodr/coco2017`](https://hf-mirror.com/datasets/phiyodr/coco2017)

- Only **captions** are used for `text-to-image` generation
- You can control the number of samples in each experiment
- Groundtruth image is available for qualitative comparison (optional)

---

## 🖼️ Sample Output Structure

