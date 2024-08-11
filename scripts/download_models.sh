#!/bin/bash

huggingface-cli download emilianJR/majicMIX_realistic_v6 --local-dir ./checkpoints/majicMIX_realistic_v6
huggingface-cli download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.safetensors --local-dir ./checkpoints/sd-vae-ft-mse
huggingface-cli download h94/IP-Adapter models/image_encoder/pytorch_model.bin models/image_encoder/config.json --local-dir ./checkpoints/ip_adapter
huggingface-cli download h94/IP-Adapter models/ip-adapter-plus_sd15.bin --local-dir ./checkpoints/ip_adapter
huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.onnx yolox_l.onnx --local-dir ./checkpoints/preprocess
huggingface-cli download lkeab/hq-sam sam_hq_vit_tiny.pth --local-dir ./checkpoints/preprocess
huggingface-cli download lllyasviel/control_v11p_sd15_openpose config.json diffusion_pytorch_model.bin --local-dir ./checkpoints/control_v11p_sd15_openpose
huggingface-cli download warmshao/EasyOOTD --local-dir ./checkpoints/easy_ootd
