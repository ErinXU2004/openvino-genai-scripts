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
    

    model_dir = Path("openvino-sd-xl-base-1.0")
    device = "GPU"

 # ==== Text-to-Image  ====
    text2image_pipe = ov_genai.Text2ImagePipeline(model_dir, device=device)
    prompt = "cute cat 4k, high-res, masterpiece, best quality, full hd, extremely detailed,  soft lighting, dynamic angle, 35mm"
    height=512
    width=512
    steps=25
    generator=ov_genai.TorchGenerator(903512)

    text_dir = Path('./generated_text2image')
    os.makedirs(text_dir, exist_ok=True)

    text_latencies = []
    num_examples = 0

    metadata_path = Path("/home/erinhua/metadata.parquet")
    if not metadata_path.exists():
        print("❌ metadata.parquet not found. Please provide it in the script directory.")
        return
    
    metadata_df = pd.read_parquet(metadata_path)
    selected_requests = metadata_df.iloc[0:num_examples].copy()

    for i,row in selected_requests.iterrows():
        prompt = row['prompt']
        clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:230]
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

   


 # ==== Image-to-Image  ====
    image2image_pipe = ov_genai.Image2ImagePipeline(model_dir, device=device)
    photo_prompt = "professional photo of a cat, extremely detailed, hyper realistic, best quality, full hd"
    photo_dir = Path("./generated_image2image")
    os.makedirs(photo_dir, exist_ok=True)
    photo_seed = 42
    num_steps = 35
    def image_to_tensor(image: Image) -> ov.Tensor:
        pic = image.convert("RGB")
        image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
        return ov.Tensor(image_data)

    image2image_latencies = []
    text_image_files = sorted(Path(text_dir).glob("*.png"))[:200]
    for i, image_path in enumerate(text_image_files):
        init_image = Image.open(image_path)
        init_tensor = image_to_tensor(init_image)
        gen = ov_genai.TorchGenerator(photo_seed + i)
        
        pbar = tqdm(total=num_steps)

        def callback(step, total_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False

        start = time.time()
        photo_result = image2image_pipe.generate(
            photo_prompt,
            image=init_tensor,
            num_inference_steps=num_steps,
            strength=0.75,
            generator=gen,
            callback=callback
        )
        end = time.time()
        image2image_latencies.append(end - start)
        out_name = image_path.stem + "_2.png"
        photo_image = Image.fromarray(photo_result.data[0])
        photo_image.save(photo_dir / out_name)
        pbar.close()
    print(f"✅ Image2Image done. Avg latency: {sum(image2image_latencies)/len(image2image_latencies):.2f}s")


if __name__ == "__main__":
    main()
