def main():
    import os
    import re
    import sys
    import time
    import requests
    from pathlib import Path
    from tqdm import tqdm
    from PIL import Image
    import openvino_genai as ov_genai
    from datasets import load_dataset
    import torch
    from diffusers import StableDiffusion3Pipeline

    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    out_dir = Path("./sd3m_fp16_generated_images")
    out_dir.mkdir(exist_ok=True)
    
    height = 512
    width = 512
    seed = 42
    max_samples = 300
    num_inference_steps = 28
    guidance_scale = 7.0

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    device = "gpu"
    pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Create folders
    gen_dir = Path("./v1.5_int4_generated_images")
    gen_dir.mkdir(exist_ok=True)

    # Load dataset
    print("🔍 Loading dataset...")
    ds = load_dataset("phiyodr/coco2017", split="train")

    latencies = []
    count = 0

    for row in tqdm(ds, desc="📦 Processing dataset"):
        if count >= max_samples:
            break

        # Check caption
        captions = row.get("captions")
        if not captions or not isinstance(captions, list) or not captions[0]:
            continue

        prompt = captions[0]
        clean_prompt = re.sub(r"[^\w\-_\.]", "_", prompt)[:100]

        gen_image_path = gen_dir / f"{clean_prompt}_{count}.png"        

        # Generate image
        pbar = tqdm(total=num_inference_steps, desc=f"🖼️ Generating {count}")

        def on_step_end(pipe,step: int, timestep: int, callback_kwargs: dict):
                pbar.update(1)
                return callback_kwargs

        start_time = time.time()
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            negative_prompt="",
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            callback_on_step_end=on_step_end,
        )
        end_time = time.time()
        pbar.close()

        latency = end_time - start_time
        latencies.append(latency)

        image = result.images[0]
        image.save(gen_image_path)

        count += 1

    print(f"\n✅ Done! Generated {count} images.")
    print(f"🕒 Avg Latency: {sum(latencies) / len(latencies):.2f} seconds")


if __name__ == "__main__":
    main()
