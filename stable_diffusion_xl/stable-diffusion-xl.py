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

    model_dir = Path("ov_sdxl")
    device = "GPU"

 # ==== Text-to-Image  ====
    text2image_pipe = ov_genai.Text2ImagePipeline(model_dir, device=device)
  #  prompt = "cute cat 4k, high-res, masterpiece, best quality, full hd, extremely detailed,  soft lighting, dynamic angle, 35mm"
    height=512
    width=512
    steps=25
    generator=ov_genai.TorchGenerator(903512)

    gen_dir = Path('./sdxl_int8_images')
    os.makedirs(text_dir, exist_ok=True)

    latencies = []
    num_examples = 300
    count = 0

    # Load dataset
    print("🔍 Loading dataset...")
    ds = load_dataset("phiyodr/coco2017", split="train")


    for row in tqdm(ds, desc="📦 Processing dataset"):
        if count >= max_samples:
            break
            
            captions = row.get("captions")
            if not captions or not isinstance(captions, list) or not captions[0]:
                continue
                
            prompt = captions[0]
            clean_prompt = re.sub(r"[^\w\-_\.]", "_", prompt)[:100]
            gen_image_path = gen_dir / f"{clean_prompt}_{count}.png"
            
            pbar = tqdm(total=num_inference_steps, desc=f"🖼️ Generating {count}")

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
            end_time = time.time()
            pbar.close()
    
            latency = end_time - start_time
            latencies.append(latency)
    
            final_image = Image.fromarray(result.data[0])
            final_image.save(gen_image_path)
    
            count += 1

         print(f"\n✅ Done! Generated {count} images.")
         print(f"🕒 Avg Latency: {sum(latencies) / len(latencies):.2f} seconds")



if __name__ == "__main__":
    main()
