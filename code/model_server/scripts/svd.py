from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch
from PIL import Image


def generate_video(image_path: str, out_path: str):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    image = Image.open(image_path)
    image = image.convert("RGB").resize((1024, 576))

    frames = pipe(
        image, decode_chunk_size=8, motion_bucket_id=180, noise_aug_strength=0.1
    ).frames[0]
    export_to_video(frames, out_path, fps=7)
