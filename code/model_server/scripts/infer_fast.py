from pathlib import Path

import torch
import os
from compel import Compel, ReturnedEmbeddingsType

# from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers import AutoencoderKL

from DeepCache import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline

from PIL import Image


def image_grid(imgs, rows, cols, resize=256):
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def infer_lora_fast(
    prompt: str,
    base_model: str,
    pretrained_vae: str,
    output_dir: str,
    lora_weights: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    vae = AutoencoderKL.from_pretrained(pretrained_vae)
    vae.to("cuda", dtype=torch.float16)
    pipe = DeepCacheStableDiffusionXLPipeline.from_pretrained(
        base_model, vae=vae, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    if lora_weights:
        pipe.load_lora_weights(lora_weights)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )

    conditioning, pooled = compel([prompt] * 4)

    images = pipe(
        # old
        num_inference_steps=25,
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        # new
        cache_interval=3,
        cache_layer_id=0,
        cache_block_id=0,
        uniform=True,
        # output_type="pt",
        # return_dict=True,
    ).images

    g = image_grid(images, rows=2, cols=2)

    print(f"outputs saved in {output_dir}")

    g.save(output_dir / "out_grid.png")

    for idx, image in enumerate(images):
        image.save(output_dir / f"out_{idx}.png")
