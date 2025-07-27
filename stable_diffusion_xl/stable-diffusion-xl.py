def main():
    from pathlib import Path
    import openvino_genai as ov_genai
    import gc
    from PIL import Image
    import time
    import os
    import re
    import requests
    import numpy as np
    import openvino as ov
    from notebook_utils import collect_telemetry, device_widget
    import openvino_genai as ov_genai
    import ipywidgets as widgets
    from gradio_helper import make_demo, make_demo_sd_xl_text2image
    from cmd_helper import optimum_cli
    import huggingface_hub as hf_hub

    

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    model_dir = Path("openvino-sd-xl-base-1.0")
    hf_hub.snapshot_download(model_id, local_dir=model_path)
    device = "GPU"

 # ==== Text-to-Image  ====
    text2image_pipe = ov_genai.Text2ImagePipeline(model_dir, device=device)
    prompt = "cute cat 4k, high-res, masterpiece, best quality, full hd, extremely detailed,  soft lighting, dynamic angle, 35mm"
    height=512
    width=512
    num_inference_steps=25
    generator=ov_genai.TorchGenerator(903512)

    text_dir = './generated_text2image'
    os.makedirs(text_dir, exist_ok=True)

    text_latencies = []
    num_examples = 200

    for i in range(num_samples):
        prompt = row['prompt']
        clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:230]
        image_path = f"{text_dir}/{clean_prompt}.png"

        # 🟢 Create new progress bar for each request
        pbar = tqdm(total=num_inference_steps)

        def callback(step, num_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False

        text_result = text2image_pipe.generate(
            prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=gen
        )
        end = time.time()
        text_latencies.append(end - start)
        image = Image.fromarray(text_result.data[0])
        image.save(image_path)
        pbar.close()

    print(f"✅ Text2Image done. Avg latency: {sum(text_latencies)/len(text_latencies):.2f}s")

   


 # ==== Image-to-Image  ====
    image2image_pipe = ov_genai.Image2ImagePipeline(model_dir, device=device.value)
    photo_prompt = "professional photo of a cat, extremely detailed, hyper realistic, best quality, full hd"

    def image_to_tensor(image: Image) -> ov.Tensor:
        pic = image.convert("RGB")
        image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
        return ov.Tensor(image_data)

    image2image_latencies = []
    for i in range(num_samples):
        init_image = Image.open(text_dir / f"cat_text2img_{i}.png")
        init_tensor = image_to_tensor(init_image)
        gen = ov_genai.TorchGenerator(photo_seed + i)
        start = time.time()
        photo_result = image2image_pipe.generate(
            photo_prompt,
            image=init_tensor,
            num_inference_steps=35,
            strength=0.75,
            generator=gen
        )
        end = time.time()
        image2image_latencies.append(end - start)
        photo_image = Image.fromarray(photo_result.data[0])
        photo_image.save(photo_dir / f"cat_photo_{i}.png")

    print(f"✅ Image2Image done. Avg latency: {sum(image2image_latencies)/len(image2image_latencies):.2f}s")


