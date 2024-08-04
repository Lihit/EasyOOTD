# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : EasyOOTD
# @FileName: app.py
import pdb

import gradio as gr
import os
import random
import torch
import numpy as np
import torch
from segment_anything_hq import sam_model_registry, SamPredictor
from PIL import Image
from PIL import ImageDraw
import cv2
from easy_ootd.models.yolox_det_model import YoloxDetModel
from easy_ootd.models.dwpose_model import DWPoseModel
from easy_ootd.common.draw import draw_pose_v2
from easy_ootd.models.unet_2d_reference import UNet2DReferenceModel
from easy_ootd.pipelines.easy_ootd_pipeline import EasyOOTDPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    LCMScheduler,
    DPMSolverMultistepScheduler
)
import os
from PIL import Image
from omegaconf import OmegaConf

example_path = os.path.join(os.path.dirname(__file__), 'assets/app_examples')
cfg_path = "configs/inference.yaml"
cfg = OmegaConf.load(cfg_path)

weight_dtype = torch.float16

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"device:{device}")
sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_model_path)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)

yolox_model = YoloxDetModel(model_path=cfg.det_model_path)
pose_model = DWPoseModel(model_path=cfg.pose_model_path)

# unet
unet = UNet2DReferenceModel.from_pretrained(
    cfg.base_model_path,
    subfolder="unet",
    unet_use_reference_attention=True,
)
unet.load_ip_adapter(cfg.ip_adapter_path)
unet.load_reference_adapter(cfg.ootd_adapter_path)
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

lcm_scheduler = LCMScheduler.from_pretrained(cfg.base_model_path,
                                             subfolder="scheduler",
                                             local_files_only=True)
dpm_scheduler = DPMSolverMultistepScheduler.from_config(
    cfg.base_model_path,
    use_karras_sigmas=True,
    algorithm_type='sde-dpmsolver++',
    subfolder="scheduler",
    local_files_only=True
)
pipe = EasyOOTDPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    controlnet=controlnet,
    scheduler=dpm_scheduler,
    unet=unet,
    image_encoder=image_enc,
    lcm_lora_path=cfg.lcm_lora_path
)

# 模特的全局信息
model_img_path_g = ''
model_img_g = None
model_points_g = []
model_plabels_g = []
model_img_new = True
model_sam_features_g = None
model_sam_interm_features_g = None
model_sam_original_size = None
model_sam_input_size = None
model_sam_mask_g = None

# 衣服的全局信息
garment_img_path_g = ''
garment_img_g = None
garment_points_g = []
garment_plabels_g = []
garment_img_new = True
garment_sam_features_g = None
garment_sam_interm_features_g = None
garment_sam_original_size = None
garment_sam_input_size = None
garment_sam_mask_g = None

# 姿态的全局信息
pose_img_path_g = ''
pose_img_g = None


def get_sam_features(image):
    """
    获取到
    :param image:
    :return:
    """
    global predictor
    image = np.array(image)
    input_image = predictor.transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=predictor.device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    sam_original_size = image.shape[:2]
    sam_input_size = tuple(input_image_torch.shape[-2:])
    input_image = predictor.model.preprocess(input_image_torch)
    sam_features, sam_interm_features = predictor.model.image_encoder(input_image)
    return sam_features, sam_interm_features, sam_original_size, sam_input_size


def sam_predict(point_coords, point_labels, features, interm_features, input_size, original_size, mask_input=None,
                box=None):
    global predictor
    # Transform input prompts
    coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
    if point_coords is not None:
        point_coords = predictor.transform.apply_coords(point_coords, original_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    if box is not None:
        box = predictor.transform.apply_boxes(box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
        box_torch = box_torch[None, :]
    if mask_input is not None:
        mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float32, device=device)
        mask_input_torch = mask_input_torch[None, :, :, :]

    if point_coords is not None:
        points = (coords_torch, labels_torch)
    else:
        points = None

    # Embed prompts
    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
        points=points,
        boxes=box_torch,
        masks=mask_input_torch,
    )

    # Predict masks
    low_res_masks, iou_predictions = predictor.model.mask_decoder(
        image_embeddings=features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        hq_token_only=False,
        interm_embeddings=interm_features,
    )

    # Upscale the masks to the original image resolution
    masks = predictor.model.postprocess_masks(low_res_masks, input_size, original_size)
    masks = masks > predictor.model.mask_threshold
    masks_np = masks[0].detach().cpu().numpy()
    iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
    low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
    return masks_np, iou_predictions_np, low_res_masks_np


def update_model_info(model_img_path):
    global model_img_g, model_points_g, model_plabels_g, model_img_new, model_sam_features_g, model_sam_interm_features_g, \
        model_sam_original_size, model_sam_input_size, model_sam_mask_g, model_img_path_g
    if model_img_path is None:
        model_img_g = None
        model_points_g = []
        model_plabels_g = []
        model_sam_features_g, model_sam_interm_features_g, model_sam_original_size, model_sam_input_size = None, None, None, None
        return
    if not model_img_path.endswith(".webp") and model_img_path != model_img_path_g:
        model_img = Image.open(model_img_path)
        model_img_g = model_img.copy()
        model_points_g = []
        model_plabels_g = []
        model_sam_features_g, model_sam_interm_features_g, model_sam_original_size, model_sam_input_size = get_sam_features(
            model_img_g)
        model_img_path_g = model_img_path
        print(f"更新模特图片: {model_img_path_g}, 图片尺寸为:", model_img_g.size)


def update_garment_info(garment_img_path):
    global garment_img_g, garment_points_g, garment_plabels_g, garment_img_new, garment_sam_features_g, garment_sam_interm_features_g, \
        garment_sam_original_size, garment_sam_input_size, garment_sam_mask_g, garment_img_path_g
    if garment_img_path is None:
        garment_img_g = None
        garment_points_g = []
        garment_plabels_g = []
        garment_sam_features_g, garment_sam_interm_features_g, garment_sam_original_size, garment_sam_input_size = None, None, None, None
        return
    if not garment_img_path.endswith(".webp") and garment_img_path != garment_img_path_g:
        garment_img = Image.open(garment_img_path)
        garment_img_g = garment_img.copy()
        garment_points_g = []
        garment_plabels_g = []
        garment_sam_features_g, garment_sam_interm_features_g, garment_sam_original_size, garment_sam_input_size = get_sam_features(
            garment_img_g)
        garment_img_path_g = garment_img_path
        print(f"更新服装图片: {garment_img_path_g}, 图片尺寸为:", garment_img_g.size)


def update_pose_info(pose_img_path):
    global pose_img_path_g, pose_img_g
    pose_img_path_g = pose_img_path
    if pose_img_path is None:
        pose_img_g = None
        print("已经删除姿态图片")
    else:
        pose_img = Image.open(pose_img_path_g)
        pose_img_g = pose_img.copy()
        print(f"更新姿态图片: {model_img_path_g}, 图片尺寸为:", pose_img_g.size)


def update_lcm(use_lcm):
    global pipe, dpm_scheduler, lcm_scheduler, cfg
    lcm_lora_path = cfg.lcm_lora_path
    if use_lcm:
        print("change to using LCM Lora")
        pipe.unet.load_lora_weights(lcm_lora_path)
        pipe.scheduler = lcm_scheduler
    else:
        print("change to not using LCM Lora")
        pipe.unet.unload_lora_weights()
        pipe.scheduler = dpm_scheduler


def blend_with_mask(image, mask, blend_color=(255, 102, 102), blend_alpha=0.5):
    """
    Blends the given image with the blend_color where the mask is 1.

    Parameters:
    - image: PIL Image object
    - mask: numpy array with the same width and height as the image, containing 0 or 1
    - blend_color: tuple, the color to blend with (default is light red (255, 102, 102))
    - blend_alpha: float, the alpha value of the blend color (0 is fully transparent, 1 is fully opaque)

    Returns:
    - blended_image: PIL Image object with the blended color applied
    """
    # Convert image to numpy array
    image_np = np.array(image).astype(np.float32)

    # Create blend color array with the same shape as image_np
    blend_color_np = np.array(blend_color).astype(np.float32)

    # Normalize mask to have the same shape as the image channels
    mask_expanded = np.expand_dims(mask, axis=-1)

    # Apply blending
    image_np = (1 - mask_expanded * blend_alpha) * image_np + mask_expanded * blend_alpha * blend_color_np

    # Ensure values are in the valid range
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    blended_image = Image.fromarray(image_np)

    return blended_image


def get_points_with_draw_on_model(image, label, evt: gr.SelectData):
    global model_img_g, model_points_g, model_plabels_g, model_img_new, model_sam_features_g, \
        model_sam_interm_features_g, model_sam_original_size, model_sam_input_size, model_sam_mask_g
    if model_sam_features_g is not None:
        x, y = evt.index[0], evt.index[1]
        model_points_g.append([x, y])
        model_plabels_g.append(1 if label == "add" else 0)

        print(x, y, label == "add")
        # 创建一个可以在图像上绘图的对象
        image_copy = model_img_g.copy()
        w, h = image_copy.size
        prev_mask = None
        for i in range(len(model_points_g)):
            masks_in = prev_mask[None] if i > 0 else None
            masks, scores, logits = sam_predict(np.array(model_points_g)[:i + 1], np.array(model_plabels_g)[:i + 1],
                                                model_sam_features_g,
                                                model_sam_interm_features_g,
                                                model_sam_input_size,
                                                model_sam_original_size,
                                                masks_in, None)
            prev_mask = logits[np.argmax(scores)]
        mask = masks[0].copy()
        mask = cv2.resize(mask.astype(np.float32), (w, h))
        model_sam_mask_g = mask.copy()
        image_copy = blend_with_mask(image_copy, mask, blend_color=(128, 128, 128), blend_alpha=0.8)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(model_points_g):
            point_radius, point_color = 10, (0, 255, 0) if model_plabels_g[i] == 1 else (255, 0, 0)
            draw.ellipse(
                [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
                fill=point_color,
            )
        return image_copy
    else:
        return image


def get_points_with_draw_on_garment(image, label, evt: gr.SelectData):
    global garment_img_g, garment_points_g, garment_plabels_g, garment_img_new, garment_sam_features_g, \
        garment_sam_interm_features_g, garment_sam_original_size, garment_sam_input_size, garment_sam_mask_g
    if garment_sam_features_g is not None:
        x, y = evt.index[0], evt.index[1]
        garment_points_g.append([x, y])
        garment_plabels_g.append(1 if label == "add" else 0)

        print(x, y, label == "add")
        # 创建一个可以在图像上绘图的对象
        image_copy = garment_img_g.copy()
        w, h = image_copy.size
        prev_mask = None
        for i in range(len(garment_points_g)):
            masks_in = prev_mask[None] if i > 0 else None
            masks, scores, logits = sam_predict(np.array(garment_points_g)[:i + 1], np.array(garment_plabels_g)[:i + 1],
                                                garment_sam_features_g,
                                                garment_sam_interm_features_g,
                                                garment_sam_input_size,
                                                garment_sam_original_size,
                                                masks_in, None)
            prev_mask = logits[np.argmax(scores)]
        # masks, scores, logits = sam_predict(np.array(garment_points_g), np.array(garment_plabels_g),
        #                                     garment_sam_features_g,
        #                                     garment_sam_interm_features_g,
        #                                     garment_sam_original_size,
        #                                     garment_sam_input_size,
        #                                     None, None)
        mask = masks[0].copy()
        mask = cv2.resize(mask.astype(np.float32), (w, h))
        garment_sam_mask_g = mask.copy()
        image_copy = blend_with_mask(image_copy, 1 - mask, blend_color=(128, 128, 128), blend_alpha=0.8)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(garment_points_g):
            point_radius, point_color = 10, (0, 255, 0) if garment_plabels_g[i] == 1 else (255, 0, 0)
            draw.ellipse(
                [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
                fill=point_color,
            )
        return image_copy
    else:
        return image


def undo_draw_on_model(image):
    global model_img_g, model_points_g, model_plabels_g, model_img_new, model_sam_features_g, \
        model_sam_interm_features_g, model_sam_original_size, model_sam_input_size, model_sam_mask_g
    if model_sam_features_g is not None and len(model_points_g):
        model_points_g = model_points_g[:-1]
        model_plabels_g = model_plabels_g[:-1]
        # 创建一个可以在图像上绘图的对象
        image_copy = model_img_g.copy()
        if len(model_points_g) == 0:
            return image_copy
        w, h = image_copy.size
        prev_mask = None
        for i in range(len(model_points_g)):
            masks_in = prev_mask[None] if i > 0 else None
            masks, scores, logits = sam_predict(np.array(model_points_g)[:i + 1], np.array(model_plabels_g)[:i + 1],
                                                model_sam_features_g,
                                                model_sam_interm_features_g,
                                                model_sam_input_size,
                                                model_sam_original_size,
                                                masks_in, None)
            prev_mask = logits[np.argmax(scores)]
        mask = masks[0].copy()
        mask = cv2.resize(mask.astype(np.float32), (w, h))
        model_sam_mask_g = mask.copy()
        image_copy = blend_with_mask(image_copy, mask, blend_color=(128, 128, 128), blend_alpha=0.8)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(model_points_g):
            point_radius, point_color = 10, (0, 255, 0) if model_plabels_g[i] == 1 else (255, 0, 0)
            draw.ellipse(
                [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
                fill=point_color,
            )
        return image_copy
    else:
        return image


def clear_draw_on_model(image):
    global model_img_g, model_points_g, model_plabels_g, model_img_new, model_sam_features_g, \
        model_sam_interm_features_g, model_sam_original_size, model_sam_input_size, model_sam_mask_g
    model_points_g = []
    model_plabels_g = []
    if model_img_g is None:
        return None
    image_copy = model_img_g.copy()
    return image_copy


def undo_draw_on_garment(image):
    global garment_img_g, garment_points_g, garment_plabels_g, garment_img_new, garment_sam_features_g, \
        garment_sam_interm_features_g, garment_sam_original_size, garment_sam_input_size, garment_sam_mask_g
    if garment_sam_features_g is not None and len(garment_points_g):
        garment_points_g = garment_points_g[:-1]
        garment_plabels_g = garment_plabels_g[:-1]
        # 创建一个可以在图像上绘图的对象
        image_copy = garment_img_g.copy()
        if len(garment_points_g) == 0:
            return image_copy
        w, h = image_copy.size
        prev_mask = None
        for i in range(len(garment_points_g)):
            masks_in = prev_mask[None] if i > 0 else None
            masks, scores, logits = sam_predict(np.array(garment_points_g)[:i + 1], np.array(garment_plabels_g)[:i + 1],
                                                garment_sam_features_g,
                                                garment_sam_interm_features_g,
                                                garment_sam_input_size,
                                                garment_sam_original_size,
                                                masks_in, None)
            prev_mask = logits[np.argmax(scores)]
        mask = masks[0].copy()
        mask = cv2.resize(mask.astype(np.float32), (w, h))
        garment_sam_mask_g = mask.copy()
        image_copy = blend_with_mask(image_copy, mask, blend_color=(128, 128, 128), blend_alpha=0.8)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(garment_points_g):
            point_radius, point_color = 10, (0, 255, 0) if garment_plabels_g[i] == 1 else (255, 0, 0)
            draw.ellipse(
                [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
                fill=point_color,
            )
        return image_copy
    else:
        return image


def clear_draw_on_garment(image):
    global garment_img_g, garment_points_g, garment_plabels_g, garment_img_new, garment_sam_features_g, \
        garment_sam_interm_features_g, garment_sam_original_size, garment_sam_input_size, garment_sam_mask_g
    garment_points_g = []
    garment_plabels_g = []
    if garment_img_g is None:
        return None
    image_copy = garment_img_g.copy()
    return image_copy


def draw_arms(keypoints, height, width):
    im_arms = Image.new('L', (width, height))
    arms_draw = ImageDraw.Draw(im_arms)
    shoulder_right = tuple(keypoints[6][:2])
    shoulder_left = tuple(keypoints[5][:2])
    elbow_right = tuple(keypoints[8][:2])
    elbow_left = tuple(keypoints[7][:2])
    wrist_right = tuple(keypoints[8][:2] / 5 + 4 * keypoints[10][:2] / 5)
    wrist_left = tuple(keypoints[7][:2] / 5 + 4 * keypoints[9][:2] / 5)

    arms_draw.line(np.concatenate(
        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
        np.uint16).tolist(), 'white', 30, 'curve')

    if height > 512:
        im_arms = cv2.dilate(np.float32(im_arms), np.ones((9, 9), np.uint16), iterations=3)
    return im_arms.astype(np.uint8)


def get_bounding_rectangle_mask(h, w, pt1, pt2, pt3):
    """
    Creates a mask of size h x w with a bounding rectangle that surrounds three points where pt1 and pt2 form the longer edge.

    Parameters:
    h, w: int
        The height and width of the mask.
    pt1, pt2, pt3: tuple
        The coordinates of the three points (x, y).

    Returns:
    np.array
        The mask with the rectangle drawn on it.
    """
    # Initialize the mask with zeros
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert points to numpy arrays
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt3 = np.array(pt3)

    # Compute the midpoint of pt1 and pt2
    midpoint = (pt1 + pt2) / 2

    # Compute the vector from pt1 to pt2
    vector = pt2 - pt1

    # Normalize the vector
    vector = vector / np.linalg.norm(vector)

    # Compute the perpendicular vector
    perp_vector = np.array([-vector[1], vector[0]])

    # Compute the distance from pt3 to the line pt1-pt2
    distance = np.dot(perp_vector, pt3 - midpoint)

    # Compute the four vertices of the rectangle
    v1 = midpoint + distance * perp_vector + vector * np.linalg.norm(pt2 - pt1) / 2
    v2 = midpoint - distance * perp_vector + vector * np.linalg.norm(pt2 - pt1) / 2
    v3 = midpoint - distance * perp_vector - vector * np.linalg.norm(pt2 - pt1) / 2
    v4 = midpoint + distance * perp_vector - vector * np.linalg.norm(pt2 - pt1) / 2

    # Determine the bounding rectangle vertices
    vertices = np.array([v1, v2, v3, v4], dtype=np.int32)

    # Fill the bounding rectangle in the mask
    cv2.fillPoly(mask, [vertices], 255)

    return mask


def draw_chests(keypoints, height, width):
    lshou = keypoints[5, :2]
    rshou = keypoints[6, :2]
    lshou_ = lshou + (lshou - rshou) / 5
    rshou_ = rshou + (rshou - lshou) / 5
    neck1 = (lshou_ + rshou_) / 2
    neck = 2 * neck1 / 3 + keypoints[0, :2] / 3
    mask = get_bounding_rectangle_mask(height, width, lshou_, rshou_, neck)
    if height > 512:
        mask = cv2.dilate(np.float32(mask), np.ones((9, 9), np.uint16), iterations=3)
    return mask.astype(np.uint8)


def transform_mask(mask, point1, point2, max_y):
    # 计算两点之间的向量
    vec = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    vev_len = np.linalg.norm(vec)
    # 归一化向量
    vec = vec / vev_len
    # 计算延长线段的两个端点
    point1 = (point1[0] - vec[0] * vev_len / 3, point1[1] - vec[1] * vev_len / 3)
    point2 = (point2[0] + vec[0] * vev_len / 3, point2[1] + vec[1] * vev_len / 3)

    # 找到mask的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 寻找面积最大的子mask
    largest_cnt = max(contours, key=cv2.contourArea)

    # 计算轮廓近似，假设四边形
    epsilon = 0.02 * cv2.arcLength(largest_cnt, True)
    approx = cv2.approxPolyDP(largest_cnt, epsilon, True)

    # 对四边形顶点按照y坐标进行排序
    sorted_points = sorted(approx, key=lambda x: x[0][1])

    if len(sorted_points) < 2:
        return None

    # 确定最上方的线段的两个端点
    if sorted_points[0][0][0] < sorted_points[1][0][0]:
        top_segment = np.float32([sorted_points[0][0], sorted_points[1][0]])
    else:
        top_segment = np.float32([sorted_points[1][0], sorted_points[0][0]])

    # 计算目标线段的两个端点
    target_segment = np.float32([point1, point2])
    # 计算旋转和缩放
    scale = np.linalg.norm(target_segment[1] - target_segment[0]) / np.linalg.norm(top_segment[1] - top_segment[0])
    angle = np.arctan2(target_segment[1][1] - target_segment[0][1], target_segment[1][0] - target_segment[0][0]) - \
            np.arctan2(top_segment[1][1] - top_segment[0][1], top_segment[1][0] - top_segment[0][0])

    # 创建缩放和旋转矩阵
    M_scale = np.array([[scale, 0], [0, scale]])
    M_rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # 计算仿射变换矩阵
    M_affine = M_rotate @ M_scale
    tx = target_segment[0][0] - M_affine[0][0] * top_segment[0][0] - M_affine[0][1] * top_segment[0][1]
    ty = target_segment[0][1] - M_affine[1][0] * top_segment[0][0] - M_affine[1][1] * top_segment[0][1]
    M_affine = np.hstack((M_affine, np.array([[tx], [ty]])))

    # 对mask进行仿射变换
    transformed_mask = cv2.warpAffine(mask, M_affine, (mask.shape[1], mask.shape[0]))
    # 查找变换后的mask的最低点
    if (transformed_mask > 0).any():
        lowest_point_y = np.max(np.where(transformed_mask > 0)[0])
        # 如果最低点超过max_y，计算y方向的缩放比例
        if lowest_point_y > max_y:
            y_scale = max_y / lowest_point_y
            # 创建y方向的缩放矩阵
            M_y_scale = np.array([[1, 0], [0, y_scale]])

            # 更新仿射变换矩阵
            M_affine[:2, :2] = M_affine[:2, :2] @ M_y_scale

            # 对mask进行y方向的缩放
            transformed_mask = cv2.warpAffine(mask, M_affine, (mask.shape[1], mask.shape[0]))
    return transformed_mask


def process_model_image(model_img, model_mask, cloth_mask, model_kpts2d, target_h, target_w, dress_type, category):
    human_img = cv2.resize(model_img, (target_w, target_h))
    human_mask = cv2.resize(model_mask, (target_w, target_h))
    kpts2d = model_kpts2d.copy()
    kpt_thred = 0.4
    if dress_type and dress_type in ["dress", "skirt"]:
        if (kpts2d[[15, 16], -1] > kpt_thred).all():
            max_y = np.mean(kpts2d[[15, 16], 1]) * 0.9
        else:
            max_y = human_mask.shape[0]
        left_hip = kpts2d[11, :2]
        right_hip = kpts2d[12, :2]
        if dress_type == "dress":
            # 如果是连衣裙的话, 需要截取上半部分
            x, y, w, h = cv2.boundingRect(cv2.findNonZero(cloth_mask))
            cloth_mask[:int(y + h * 0.4)] = 0
        dress_mask = transform_mask(cloth_mask, right_hip, left_hip, max_y)
        if dress_mask is not None:
            human_mask[dress_mask > 0] = 255

    if category in ["full_body", "upper_body"]:
        h, w = human_mask.shape[:2]
        arms_mask = draw_arms(kpts2d, h, w)
        human_mask = np.clip(human_mask + arms_mask, 0, 255)
        chests_mask = draw_chests(kpts2d, h, w)
        human_mask = np.clip(human_mask + chests_mask, 0, 255)
    human_mask = cv2.dilate(human_mask, kernel=np.ones((5, 5)), iterations=3)

    contours, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if contours is None or len(contours) == 0:
        return None
    max_contour = contours[0]
    mask_ret = np.zeros(human_mask.shape[:2], np.uint8)
    cv2.drawContours(mask_ret, [max_contour], -1, 1, -1)
    human_mask[mask_ret == 1] = 255

    contours, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if contours is None or len(contours) == 0:
        return None
    max_contour = contours[0]
    mask_ret = np.zeros(human_mask.shape[:2], np.uint8)
    cv2.drawContours(mask_ret, [max_contour], -1, 1, -1)
    human_mask = human_mask * mask_ret

    human_mask = (1.0 - human_mask / 255.0)[..., None]
    human_img_gray = np.ones_like(human_img) * 128
    human_img = human_img * human_mask + human_img_gray * (1 - human_mask)
    human_img = human_img.astype(np.uint8).clip(0, 255)
    return Image.fromarray(human_img)


def process_garment_image(garment_img, garment_mask, target_h, target_w):
    from easy_ootd.models.human_parsing_model import _box2cs, get_affine_transform
    bbox = cv2.boundingRect(cv2.findNonZero(garment_mask))
    input_size = [target_w, target_h]
    c, s = _box2cs(bbox, input_size, rescale=1.1)
    r = 0
    trans = get_affine_transform(c, s, r, input_size)

    garment_img = cv2.warpAffine(garment_img, trans, (int(input_size[0]), int(input_size[1])), flags=cv2.INTER_LINEAR)
    garment_mask_ret = cv2.warpAffine(garment_mask, trans, (int(input_size[0]), int(input_size[1])),
                                      flags=cv2.INTER_LINEAR)

    garment_mask = (garment_mask_ret / 255.)[..., None]
    garment_img_gray = np.ones_like(garment_img) * 128
    garment_img = garment_img * garment_mask + garment_img_gray * (1 - garment_mask)
    garment_img = Image.fromarray(garment_img.astype(np.uint8).clip(0, 255))
    return garment_img, garment_mask_ret


def process_cond_image(pose_img):
    H, W = pose_img.shape[:2]
    det_bbox = yolox_model.predict(pose_img)
    if len(det_bbox) == 0:
        return None
    # select one
    det_bbox = det_bbox[:1]
    keypoints2d = pose_model.predict(pose_img, det_bbox)
    img_draw = draw_pose_v2(keypoints2d, H, W, ref_w=720)
    img_draw = Image.fromarray(img_draw)
    return img_draw, keypoints2d[0]


def run_outfit(n_steps, guide_scale, seed, short_size, category, dress_type, use_lcm, use_pose_detect):
    global model_img_g, model_sam_mask_g, garment_img_g, garment_sam_mask_g, pose_img_g

    assert model_img_g is not None and garment_img_g is not None

    model_img = np.array(model_img_g)
    org_h, org_w = model_img.shape[:2]
    scale = short_size / min(org_h, org_w)
    h, w = int(org_h * scale), int(org_w * scale)
    model_img = cv2.resize(model_img, (w, h))

    ## 处理 cond image
    if pose_img_g is None:
        pose_img = model_img.copy()
    else:
        pose_img = np.array(pose_img_g)
    pose_img = cv2.resize(pose_img, (w, h))
    if use_pose_detect:
        pose_img_pil, keypoints2d = process_cond_image(pose_img)
    else:
        pose_img_pil = Image.fromarray(pose_img)
        keypoints2d = None

    ## 处理 garment image
    garment_img = np.array(garment_img_g)
    garment_mask = (garment_sam_mask_g * 255).astype(np.uint8)
    garment_img_pil, garment_mask = process_garment_image(garment_img, garment_mask, h, w)

    ## 处理 model image
    model_mask = (model_sam_mask_g * 255).astype(np.uint8)
    if pose_img_g is not None or keypoints2d is None:
        det_bbox = yolox_model.predict(model_img)
        if len(det_bbox) == 0:
            return None
        # select one
        det_bbox = det_bbox[:1]
        keypoints2d = pose_model.predict(model_img, det_bbox)[0]
    model_img_pil = process_model_image(model_img, model_mask, garment_mask, keypoints2d, h, w, dress_type, category)

    generator = torch.Generator(device=device).manual_seed(seed)

    res_image_pil = pipe(
        prompt="",
        human_refer_image=model_img_pil,
        cloth_refer_image=garment_img_pil,
        condition_image=pose_img_pil,
        width=w,
        height=h,
        num_inference_steps=n_steps,
        guidance_scale=guide_scale,
        generator=generator,
        controlnet_conditioning_scale=1.0,
        negative_prompt="",
        use_lcm=use_lcm
    ).images[0]
    res_image_pil = res_image_pil.resize((org_w, org_h))
    return [res_image_pil, model_img_pil, garment_img_pil, pose_img_pil]


def run_app():
    theme = gr.themes.Soft()
    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    block = gr.Blocks(title="EasyOOTD", theme=theme, js=js_func).queue()
    with block:
        with gr.Row():
            gr.Markdown("# EasyOOTD: Pose-Controllable Virtual Try-On via Diffusion Adapter")

        with gr.Row():
            with gr.Column():
                model_dir = os.path.join(example_path, "models")
                model_paths = [os.path.join(model_dir, name) for name in os.listdir(model_dir)]

                model_img = gr.Image(label="模特(Model)", sources=['upload'], type="filepath", height=384,
                                     value=None)
                model_img.change(update_model_info, inputs=model_img)
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["add", "remove"],
                        value="add",
                        interactive=True,
                        label="SAM point prompts"
                    )
                with gr.Row():
                    model_undo = gr.Button(value="undo")
                    model_clear = gr.Button(value="clear")

                example = gr.Examples(
                    label="Examples",
                    inputs=model_img,
                    examples_per_page=10,
                    examples=model_paths
                )
                model_img.select(get_points_with_draw_on_model, [model_img, add_or_remove], model_img)
                model_undo.click(undo_draw_on_model, model_img, model_img)
                model_clear.click(clear_draw_on_model, model_img, model_img)
            with gr.Column():
                garment_dir = os.path.join(example_path, "garments")
                garment_paths = [os.path.join(garment_dir, name) for name in os.listdir(garment_dir)]

                garment_img = gr.Image(label="服装(Garment)", sources=['upload'], type="filepath", height=384,
                                       value=None)
                garment_img.change(update_garment_info, inputs=garment_img)
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["add", "remove"],
                        value="add",
                        interactive=True,
                        label="SAM point prompts"
                    )
                with gr.Row():
                    garment_undo = gr.Button(value="undo")
                    garment_clear = gr.Button(value="clear")

                example = gr.Examples(
                    label="Examples",
                    inputs=garment_img,
                    examples_per_page=10,
                    examples=garment_paths
                )
                garment_img.select(get_points_with_draw_on_garment, [garment_img, add_or_remove], garment_img)
                garment_undo.click(undo_draw_on_garment, garment_img, garment_img)
                garment_clear.click(clear_draw_on_garment, garment_img, garment_img)
                category = gr.Dropdown(label="服装种类(Garment category)",
                                       choices=["upper_body", "lower_body", "full_body"], interactive=True,
                                       value="upper_body")
                dress_type = gr.Dropdown(label="裙子类型(Dress Type)", choices=["dress", "skirt", "None"],
                                         interactive=True, value=None)
            with gr.Column():
                with gr.Row():
                    cond_img = gr.Image(label="姿态(Pose)", sources=['upload'], type="filepath", height=384,
                                        value=None)
                with gr.Row():
                    use_pose_detect = gr.Checkbox(label="use pose estimation",
                                                  info="Uncheck if you upload skeleton image",
                                                  value=True)
                with gr.Row():
                    result_gallery = gr.Gallery(label='Output', show_label=True, allow_preview=True, format="png")
                cond_img.change(update_pose_info, inputs=cond_img)

        with gr.Column():
            run_button = gr.Button(value="Run")
            n_steps = gr.Slider(label="Steps", minimum=1, maximum=25, value=20, interactive=True, step=1)
            guide_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, interactive=True,
                                    step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, interactive=True, value=1026)
            short_size = gr.Slider(label="Short Size", minimum=512, maximum=1024, value=512, interactive=True,
                                   step=32)
            use_lcm = gr.Checkbox(label="use lcm lora", info="set cfg=1.0 and steps=4 if using lcm", value=False)
            use_lcm.change(update_lcm, inputs=[use_lcm])

        run_button.click(run_outfit, inputs=[n_steps, guide_scale, seed, short_size, category, dress_type, use_lcm,
                                             use_pose_detect],
                         outputs=result_gallery)

    block.launch(server_name='127.0.0.1', server_port=7865)


if __name__ == '__main__':
    run_app()
