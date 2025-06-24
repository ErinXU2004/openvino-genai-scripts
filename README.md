# OpenVINO GenAI Text-to-Image Demos

This repository contains 4 OpenVINO-based text-to-image generation models that I have successfully tested and run on the Intel AI PC development server.

All scripts and notebooks are adapted from official [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks), and used for internal demo purposes.

## 📦 Included Models

| Folder | Model | Description |
|--------|-------|-------------|
| `stable_diffusion_xl` | Stable Diffusion XL | Large-scale text-to-image generation |
| `stable_diffusion_v3` | Stable Diffusion v3 | Optimized OpenVINO pipeline with SD v3 weights |
| `flux` | FLUX.1 Image Generation | Fast, lightweight diffusion model |
| `text_to_image` | OpenVINO Text-to-Image Demo | General GenAI demo pipeline |

## 🚀 Usage

Each folder contains a ready-to-run Jupyter Notebook (`.ipynb`) with setup and usage instructions inline.

To run these notebooks:
1. Set up a Python environment with OpenVINO installed.
2. Follow instructions in each notebook.
3. **Note**: Model weights are **excluded** from this repository due to size. Please follow instructions in the notebooks to download them if needed.

## 📁 File Structure Example
genai-scripts/
├── flux/
│ └── flux.1-image-generation.ipynb
├── stable_diffusion_v3/
│ └── stable-diffusion-v3.ipynb
├── stable_diffusion_xl/
│ └── stable-diffusion-xl.ipynb
├── text_to_image/
│ └── text-to-image-genai.ipynb
└── README.md


## 🔒 Notes

- Large files such as `.bin`, `.onnx`, `INT4/`, and virtual environments are excluded via `.gitignore`.
- The goal of this repo is to organize and share the tested model scripts, not for full deployment.

## 👩‍💻 Maintained by  
Erin Xu  
University of Michigan  
erinhua@umich.edu

