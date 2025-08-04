def main():
    import platform
    import requests
    from pathlib import Path
    import os
    import re
    import sys
    import time
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    import openvino_genai as ov_genai
    from notebook_utils import collect_telemetry
    from cmd_helper import optimum_cli
    from gradio_helper import make_demo
    from datasets import load_dataset



    model_dir = "/home/erinhua/openvino-genai-scripts/FLUX.1-schnell-int4-ov"
    # Device and pipeline
    device = "GPU"
    ov_pipe = ov_genai.Text2ImagePipeline(model_dir, device=device)

    # Inference settings
    prompt = "A cat holding a sign that says hello OpenVINO"
    height = 256
    width = 256
    seed = 42
    num_inference_steps = 4

    print("Pipeline settings")
    print(f"Input text: {prompt}")
    print(f"Image size: {height} x {width}")
    print(f"Seed: {seed}")
    print(f"Number of steps: {num_inference_steps}")

    random_generator = ov_genai.TorchGenerator(seed)

    # result = ov_pipe.generate(prompt, num_inference_steps=num_inference_steps, generator=random_generator, callback=callback, height=height, width=width)
    # final_image = Image.fromarray(result.data[0])
    # final_image.save("img.png")

    images_directory = './flux_generated_images'
    os.makedirs(images_directory, exist_ok=True)

    latencies = []
    num_examples = 200

    metadata_path = Path("/home/erinhua/metadata.parquet")
    if not metadata_path.exists():
        print("❌ metadata.parquet not found. Please provide it in the script directory.")
        return
    ds = load_dataset("lmms-lab/COCO-Caption2017", split="train")
    metadata_df = pd.read_parquet(metadata_path)
    selected_requests = ds.select(range(num_examples))
    
    for i, row in selected_requests.iterrows():
        prompt = row.get('prompt') or row['captions'][0]
        clean_prompt = safe_filename(prompt)
        image_path = f"{images_directory}/{clean_prompt}.png"

        # 🟢 Create new progress bar for each request
        pbar = tqdm(total=num_inference_steps)

        def callback(step, num_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False

        start_time = time.time()
        result = ov_pipe.generate(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=random_generator,
            callback=callback,
            height=height,
            width=width
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
