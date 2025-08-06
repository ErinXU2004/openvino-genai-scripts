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
    import huggingface_hub as hf_hub

    # Settings
    model_id = "OpenVINO/stable-diffusion-v1-5-fp16-ov"
    model_path = "stable-diffusion-v1-5-fp16-ov"
    hf_hub.snapshot_download(model_id, local_dir=model_path) 
    device = "GPU"
    ov_pipe = ov_genai.Text2ImagePipeline(model_path, device=device)
    
    height = 512
    width = 512
    seed = 42
    num_inference_steps = 28
    guidance_scale = 5
    generator = ov_genai.TorchGenerator(seed)

    # Create folders
    gen_dir = Path("./v1.5_int8_generated_images")
    gt_dir = Path("./v1.5_int8_groundtruth_images")
    gen_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)

    # Load dataset
    print("🔍 Loading dataset...")
    ds = load_dataset("phiyodr/coco2017", split="train")

    latencies = []
    count = 0
    max_samples = 300

    for row in tqdm(ds, desc="📦 Processing dataset"):
        if count >= max_samples:
            break

        # Check resolution
        #if row["width"] != 640 or row["height"] != 480:
        #    continue

        # Check caption
        captions = row.get("captions")
        if not captions or not isinstance(captions, list) or not captions[0]:
            continue

        prompt = captions[0]
        clean_prompt = re.sub(r"[^\w\-_\.]", "_", prompt)[:100]

        gen_image_path = gen_dir / f"{clean_prompt}_{count}.png"
        gt_image_path = gt_dir / f"{clean_prompt}_{count}.jpg"

        # Download groundtruth image
        image_url = f"http://images.cocodataset.org/train2017/{str(row['image_id']).zfill(12)}.jpg"
        response = requests.get(image_url)
        with open(gt_image_path, "wb") as f:
            f.write(response.content)

        # Generate image
        pbar = tqdm(total=num_inference_steps, desc=f"🖼️ Generating {count}")

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
