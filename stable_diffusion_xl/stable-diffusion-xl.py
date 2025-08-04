def main():
    from tqdm import tqdm
    import pandas as pd
    from pathlib import Path
    import openvino_genai as ov_genai
    import gc
    import sys
    from PIL import Image
    import time
    import os
    import re
    import requests
    import numpy as np
    import openvino as ov
    from notebook_utils import collect_telemetry, device_widget
    import openvino_genai as ov_genai
    from cmd_helper import optimum_cli
    import huggingface_hub as hf_hub
    from gradio_helper import make_demo_sd_xl_text2image
    from datasets import load_dataset

    model_dir = Path("openvino-sd-xl-base-1.0")
    device = "GPU"

 # ==== Text-to-Image  ====
    text2image_pipe = ov_genai.Text2ImagePipeline(model_dir, device=device)
  #  prompt = "cute cat 4k, high-res, masterpiece, best quality, full hd, extremely detailed,  soft lighting, dynamic angle, 35mm"
    height=512
    width=512
    steps=25
    generator=ov_genai.TorchGenerator(903512)

    text_dir = Path('./generated_text2image')
    os.makedirs(text_dir, exist_ok=True)

    text_latencies = []
    num_examples = 200

    ds = load_dataset("lmms-lab/COCO-Caption2017", split="train")
    selected = ds.select(range(num_examples))


    for row in selected:
        prompt = row.get('prompt') or row['captions'][0]
        clean_prompt = safe_filename(prompt)
        image_path = f"{text_dir}/{clean_prompt}.png"

        # 🟢 Create new progress bar for each request
        pbar = tqdm(total=steps)

        def callback(step, num_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False
        start = time.time()
        text_result = text2image_pipe.generate(
            prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=generator,
            callback=callback
        )
        end = time.time()
        text_latencies.append(end - start)
        image = Image.fromarray(text_result.data[0])
        image.save(image_path)
        pbar.close()

#    print(f"✅ Text2Image done. Avg latency: {sum(text_latencies)/len(text_latencies):.2f}s")


if __name__ == "__main__":
    main()
