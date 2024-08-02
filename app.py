# -*- coding: utf-8 -*-
# @Time    : 2024/8/2 20:16
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : EasyOOTD
# @FileName: app.py
import pdb

import gradio as gr
import os
import random

import numpy as np
import torch
from segment_anything_hq import sam_model_registry, SamPredictor
from PIL import Image
from PIL import ImageDraw
import cv2

example_path = os.path.join(os.path.dirname(__file__), 'assets/app_examples')

sam_checkpoint = "./checkpoints/preprocess/sam_hq_vit_tiny.pth"
model_type = "vit_tiny"

weight_dtype = torch.float32

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device:{device}")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device, dtype=weight_dtype)
sam.eval()
predictor = SamPredictor(sam)

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
        coords_torch = torch.as_tensor(point_coords, dtype=weight_dtype, device=device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    if box is not None:
        box = predictor.transform.apply_boxes(box, original_size)
        box_torch = torch.as_tensor(box, dtype=weight_dtype, device=device)
        box_torch = box_torch[None, :]
    if mask_input is not None:
        mask_input_torch = torch.as_tensor(mask_input, dtype=weight_dtype, device=device)
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
    if not model_img_path.endswith(".webp") and model_img_path != model_img_path_g:
        model_img = Image.open(model_img_path)
        model_img_g = model_img.copy()
        model_points_g = []
        model_plabels_g = []
        model_img_new = True
        model_sam_features_g, model_sam_interm_features_g, model_sam_original_size, model_sam_input_size = get_sam_features(
            model_img_g)
        model_img_path_g = model_img_path
        print(f"更新模特图片: {model_img_path_g}, 图片尺寸为:", model_img_g.size)


def update_garment_info(garment_img_path):
    global garment_img_g, garment_points_g, garment_plabels_g, garment_img_new, garment_sam_features_g, garment_sam_interm_features_g, \
        garment_sam_original_size, garment_sam_input_size, garment_sam_mask_g, garment_img_path_g
    if not garment_img_path.endswith(".webp") and garment_img_path != garment_img_path_g:
        garment_img = Image.open(garment_img_path)
        garment_img_g = garment_img.copy()
        garment_points_g = []
        garment_plabels_g = []
        garment_img_new = True
        garment_sam_features_g, garment_sam_interm_features_g, garment_sam_original_size, garment_sam_input_size = get_sam_features(
            garment_img_g)
        garment_img_path_g = garment_img_path
        print(f"更新服装图片: {garment_img_path_g}, 图片尺寸为:", garment_img_g.size)


def update_pose_info(pose_img_path):
    global pose_img_path_g, pose_img_g
    pose_img_path_g = pose_img_path
    pose_img = Image.open(pose_img_path_g)
    pose_img_g = pose_img.copy()
    print(f"更新姿态图片: {model_img_path_g}, 图片尺寸为:", pose_img_g.size)


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
        point_radius, point_color = 15, (0, 255, 0) if label == "add" else (
            255,
            0,
            0,
        )
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
                                                model_sam_original_size,
                                                model_sam_input_size,
                                                masks_in, None)
            prev_mask = logits[np.argmax(scores)]
        # masks, scores, logits = sam_predict(np.array(model_points_g), np.array(model_plabels_g),
        #                                     model_sam_features_g,
        #                                     model_sam_interm_features_g,
        #                                     model_sam_original_size,
        #                                     model_sam_input_size,
        #                                     None, None)
        mask = masks[0].copy()
        mask = cv2.resize(mask.astype(np.float32), (w, h))
        model_sam_mask_g = mask.copy()
        image_copy = blend_with_mask(image_copy, mask, blend_color=(128, 128, 128), blend_alpha=1)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(model_points_g):
            point_radius, point_color = 15, (0, 255, 0) if model_plabels_g[i] == 1 else (255, 0, 0)
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
                                                garment_sam_original_size,
                                                garment_sam_input_size,
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
        image_copy = blend_with_mask(image_copy, 1 - mask, blend_color=(128, 128, 128), blend_alpha=1)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(garment_points_g):
            point_radius, point_color = 15, (0, 255, 0) if garment_plabels_g[i] == 1 else (255, 0, 0)
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
                                                model_sam_original_size,
                                                model_sam_input_size,
                                                masks_in, None)
            prev_mask = logits[np.argmax(scores)]
        # masks, scores, logits = sam_predict(np.array(model_points_g), np.array(model_plabels_g),
        #                                     model_sam_features_g,
        #                                     model_sam_interm_features_g,
        #                                     model_sam_original_size,
        #                                     model_sam_input_size,
        #                                     None, None)
        mask = masks[0].copy()
        mask = cv2.resize(mask.astype(np.float32), (w, h))
        model_sam_mask_g = mask.copy()
        image_copy = blend_with_mask(image_copy, mask, blend_color=(128, 128, 128), blend_alpha=1)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(model_points_g):
            point_radius, point_color = 15, (0, 255, 0) if model_plabels_g[i] == 1 else (255, 0, 0)
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
                                                garment_sam_original_size,
                                                garment_sam_input_size,
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
        image_copy = blend_with_mask(image_copy, mask, blend_color=(128, 128, 128), blend_alpha=1)
        draw = ImageDraw.Draw(image_copy)
        for i, (x, y) in enumerate(garment_points_g):
            point_radius, point_color = 15, (0, 255, 0) if garment_plabels_g[i] == 1 else (255, 0, 0)
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
    image_copy = garment_img_g.copy()
    return image_copy


def run_app():
    block = gr.Blocks(title="EasyOOTD").queue()
    with block:
        with gr.Row():
            gr.Markdown("# EasyOOTD: Make Virtual Try-On Easier")

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
                        label="add or remove point for SAM"
                    )
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
                        label="add or remove point for SAM"
                    )
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
                    cond_img.change(update_pose_info, inputs=cond_img)
                with gr.Row():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True,
                                                allow_preview=True, interactive=False, scale=1)

        with gr.Column():
            run_button_dc = gr.Button(value="Run")
            n_steps_dc = gr.Slider(label="Steps", minimum=1, maximum=10, value=4, interactive=True, step=1)
            image_scale_dc = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=1.0, interactive=True,
                                       step=0.1)
            seed_dc = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, interactive=True, value=-1)
            short_size_dc = gr.Slider(label="Short Size", minimum=256, maximum=768, value=512, interactive=True,
                                      step=32)

        run_button_dc.click()

    block.launch(server_name='127.0.0.1', server_port=7865)


if __name__ == '__main__':
    run_app()
