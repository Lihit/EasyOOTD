# -*- coding: utf-8 -*-
import pdb
import time

import torch
from omegaconf import OmegaConf


def test_easy_ootd_pipeline():
    from easy_ootd.models.unet_2d_reference import UNet2DReferenceModel
    from easy_ootd.pipelines.easy_ootd_pipeline import EasyOOTDPipeline
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        LCMScheduler
    )
    import os
    from PIL import Image

    weight_dtype = torch.float16

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg_path = "configs/inference.yaml"
    cfg = OmegaConf.load(cfg_path)

    # unet
    unet = UNet2DReferenceModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
        unet_use_reference_attention=True,
    )
    unet.load_ip_adapter(cfg.ip_adapter_path)
    unet.load_reference_adapter(cfg.ootd_adapter_path)
    unet.load_lora_weights(cfg.lcm_lora_path)
    unet.register_reference_hooks(reference_scale=1.0, fusion_blocks="full")
    unet = unet.to(device, dtype=weight_dtype).eval()

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(device, dtype=weight_dtype).eval()

    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_path, local_files_only=True).to(
        device, dtype=weight_dtype
    ).eval()

    tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(cfg.base_model_path, subfolder="text_encoder",
                                                 local_files_only=True).to(device, dtype=weight_dtype).eval()

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path, local_files_only=True).to(
        device, dtype=weight_dtype
    ).eval()

    scheduler = LCMScheduler.from_pretrained(cfg.base_model_path,
                                             subfolder="scheduler",
                                             local_files_only=True)

    pipe = EasyOOTDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        controlnet=controlnet,
        scheduler=scheduler,
        unet=unet,
        image_encoder=image_enc,
        lcm_lora_path=cfg.lcm_lora_path
    )
    seed = 1111
    generator = torch.Generator(device=device).manual_seed(seed)

    test_dir = "./assets/test_examples/003"
    human_ref_image_path = os.path.join(test_dir, "human_ref_imgs.png")
    cloth_ref_image_path = os.path.join(test_dir, "cloth_ref_imgs.png")
    pose_image_path = os.path.join("./assets/test_examples/003", "con_imgs.png")
    gt_image_path = os.path.join(test_dir, "gt_imgs.png")
    prompt = ""
    negative_prompt = ""
    human_ref_image_pil = Image.open(human_ref_image_path).convert("RGB")
    cloth_ref_image_pil = Image.open(cloth_ref_image_path).convert("RGB")
    pose_image_pil = Image.open(pose_image_path).convert("RGB")
    gt_image_pil = Image.open(gt_image_path).convert("RGB")

    for _ in range(1):
        t0 = time.time()
        res_image_pil = pipe(
            prompt=prompt,
            human_refer_image=human_ref_image_pil,
            cloth_refer_image=cloth_ref_image_pil,
            condition_image=pose_image_pil,
            width=512,
            height=768,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=generator,
            controlnet_conditioning_scale=1.0,
            negative_prompt=negative_prompt
        ).images[0]
        print(time.time() - t0)
    w, h = res_image_pil.size
    canvas = Image.new("RGB", (w * 4, h), "white")
    gt_image_pil = gt_image_pil.resize((w, h))
    human_ref_image_pil = human_ref_image_pil.resize((w, h))
    cloth_ref_image_pil = cloth_ref_image_pil.resize((w, h))
    canvas.paste(gt_image_pil, (0, 0))
    canvas.paste(human_ref_image_pil, (w, 0))
    canvas.paste(cloth_ref_image_pil, (w * 2, 0))
    canvas.paste(res_image_pil, (w * 3, 0))
    canvas.save(f"./results/easy_ootd_{os.path.basename(test_dir)}_{seed}.png")
    print(f"./results/easy_ootd_{os.path.basename(test_dir)}_{seed}.png")


if __name__ == '__main__':
    test_easy_ootd_pipeline()
