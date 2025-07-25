def main():
    import platform
    import subprocess
    import requests
    from pathlib import Path
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

 
    pt_pipeline_options, model_selector, load_t5, to_compress = get_pipeline_options()
    model_id = model_selector.value
    print(f"Selected {model_id} model")

    base_model_path = Path(model_id.split("/")[-1])
    model_path = base_model_path / ("FP16" if not to_compress.value else "INT4")

    if not to_compress.value:
        additional_args = {"weight-format": "fp16"}
    else:
        additional_args = {"weight-format": "int4", "group-size": "64", "ratio": "1.0"}

    if not model_path.exists():
        optimum_cli(model_id, model_path)

        device = "GPU"

    # Initialize pipeline
    if not load_t5.value:
        ov_pipe = init_pipeline_without_t5(model_path, device)
    else:
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
            callback=None
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
