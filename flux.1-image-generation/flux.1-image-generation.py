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

    # Telemetry
    collect_telemetry("flux.1-image-generation.py")

    # Model selection
    model_id = "black-forest-labs/FLUX.1-schnell"
    use_preconverted = True
    to_compress = True

    model_base_dir = Path(model_id.split("/")[-1])
    additional_args = {}

    if to_compress:
        model_dir = model_base_dir / "INT4"
        additional_args.update({"weight-format": "int4", "group-size": "64", "ratio": "1.0"})
    else:
        model_dir = model_base_dir / "FP16"
        additional_args.update({"weight-format": "fp16"})

    if not model_dir.exists():
        if not use_preconverted:
            optimum_cli(model_id, model_dir, additional_args=additional_args)
        else:
            print(f"Please manually download the model from HuggingFace: OpenVINO/{model_id.split('/')[-1]}-{model_dir.name.lower()}-ov")
            return

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
    pbar = tqdm(total=num_inference_steps)

    def callback(step, num_steps, latent):
        if num_steps != pbar.total:
            pbar.reset(num_steps)
        pbar.update(1)
        sys.stdout.flush()
        return False

    images_directory = './flux_generated_images'
    os.makedirs(images_directory, exist_ok=True)

    latencies = []
    num_examples = 200

    metadata_path = Path("metadata.parquet")
    if not metadata_path.exists():
        print("❌ metadata.parquet not found. Please provide it in the script directory.")
        return

    metadata_df = pd.read_parquet(metadata_path)
    selected_requests = metadata_df.iloc[0:num_examples].copy()
    for i, row in selected_requests.iterrows():
        prompt = row['prompt']
        clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:230]
        image_path = f"{images_directory}/{clean_prompt}.png"
        start_time = time.time()
        result = ov_pipe.generate(prompt, num_inference_steps=num_inference_steps, generator=random_generator, callback=callback, height=height, width=width)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        final_image = Image.fromarray(result.data[0])
        final_image.save(image_path)

    print(f"\n🕒 Avg Latency: {sum(latencies)/len(latencies):.2f} seconds")
    pbar.close()

    # Launch Gradio demo
    demo = make_demo(ov_pipe, model_name=str(model_base_dir))
    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)


if __name__ == "__main__":
    main()
