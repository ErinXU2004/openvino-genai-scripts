def main():
    import platform
    import subprocess
    from tqdm import tqdm
    import requests
    from pathlib import Path
    import pandas as pd
    import os
    import re
    import sys
    import time
    from PIL import Image
    import openvino_genai as ov_genai
    from notebook_utils import collect_telemetry, device_widget
    from cmd_helper import optimum_cli
    from gradio_helper import make_demo
    from sd3_helper import get_pipeline_options, init_pipeline_without_t5
    import huggingface_hub as hf_hub
 
    model_id = "/home/erinhua/openvino-genai-scripts/FLUX.1-schnell-int4-ov"
   # model_id = "OpenVINO/stable-diffusion-v1-5-fp16-ov"
    model_path = "stable-diffusion-v1-5-fp16-ov"
    hf_hub.snapshot_download(model_id, local_dir=model_path) 	    
    device = "GPU"
    ov_pipe = ov_genai.Text2ImagePipeline(model_path, device=device)

   # print("Pipeline initialized successfully.")

    # Inference settings
    prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors"
    height = 512
    width = 512
    seed = 42
    num_inference_steps = 28
    guidance_scale = 5 if "turbo" not in model_id else 0.5
    generator = ov_genai.TorchGenerator(seed)

    print("Pipeline settings")
    print(f"Input text: {prompt}")
    print(f"Image size: {height} x {width}")
    print(f"Seed: {seed}")
    print(f"Number of steps: {num_inference_steps}")


    images_directory = './SDv3_generated_images'
    os.makedirs(images_directory, exist_ok=True)
    latencies = []
    num_examples = 200

    metadata_path = Path("/home/erinhua/metadata.parquet")
    if not metadata_path.exists():
        print("❌ metadata.parquet not found. Please provide it in the script directory.")
        return
    
    metadata_df = pd.read_parquet(metadata_path)
    selected_requests = metadata_df.iloc[0:num_examples].copy()
    for i, row in selected_requests.iterrows():
        prompt = row['prompt']
        clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:230]
        image_path = f"{images_directory}/{clean_prompt}.png"

        # 🟢 Create new progress bar for each request
        pbar = tqdm(total=num_inference_steps)

        def callback(step, num_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False

        start_time = time.time()
        result = ov_pipe.generate(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback=callback
        )
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)

        final_image = Image.fromarray(result.data[0])
        final_image.save(image_path)

        # 🔴 Close pbar after use
        pbar.close()

    print(f"\n🕒 Avg Latency: {sum(latencies)/len(latencies):.2f} seconds")


if __name__ == "__main__":
    main()
