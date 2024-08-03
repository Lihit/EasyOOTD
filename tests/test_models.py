# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : EasyOOTD
# @FileName: test_models.py
import os
import pdb


def test_yolox_model():
    import cv2
    from easy_ootd.models.yolox_det_model import YoloxDetModel

    yolox_model = YoloxDetModel(model_path="./checkpoints/preprocess/yolox_l.onnx")
    img_path = "assets/app_examples/models/img_7.png"

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    det_bbox = yolox_model.predict(image)
    print(det_bbox)


def test_pose_model():
    import cv2
    from easy_ootd.models.yolox_det_model import YoloxDetModel
    from easy_ootd.models.dwpose_model import DWPoseModel
    from easy_ootd.common.draw import draw_pose_v2

    yolox_model = YoloxDetModel(model_path="./checkpoints/preprocess/yolox_l.onnx")
    pose_model = DWPoseModel(model_path="./checkpoints/preprocess/dw-ll_ucoco_384.onnx")
    img_path = "assets/app_examples/models/img_7.png"

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    det_bbox = yolox_model.predict(image)
    print(det_bbox)
    pose_ret = pose_model.predict(image, det_bbox)
    print(pose_ret)
    pdb.set_trace()
    img_draw = draw_pose_v2(pose_ret[0], pose_ret[1], H, W, ref_w=720)
    cv2.imwrite(f"results/{os.path.basename(img_path)}-pose.png", img_draw)


def test_parsing_model():
    import cv2
    import numpy as np
    from PIL import Image
    from easy_ootd.models.human_parsing_model import HumanParsingModel
    from easy_ootd.models.yolox_det_model import YoloxDetModel
    from easy_ootd.models.dwpose_model import DWPoseModel
    from easy_ootd.common.draw import draw_pose_v2
    from easy_ootd.common.draw import get_palette

    label_map = {
        "background": 0,
        "hat": 1,
        "hair": 2,
        "sunglasses": 3,
        "upper_clothes": 4,
        "skirt": 5,
        "pants": 6,
        "dress": 7,
        "belt": 8,
        "left_shoe": 9,
        "right_shoe": 10,
        "head": 11,
        "left_leg": 12,
        "right_leg": 13,
        "left_arm": 14,
        "right_arm": 15,
        "bag": 16,
        "scarf": 17,
    }

    yolox_model = YoloxDetModel(model_path="./checkpoints/preprocess/yolox_l.onnx")
    parsing_model = HumanParsingModel(model_path="./checkpoints/preprocess/checkpoints/humanparsing/parsing_atr.onnx")

    img_path = "assets/app_examples/garments/img_3.png"

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    det_bbox = yolox_model.predict(image)
    if len(det_bbox) == 0:
        det_bbox = [0, 0, W, H]
    else:
        det_bbox = det_bbox[0]
        det_bbox = [det_bbox[0], det_bbox[1], det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]]
    parse_maps = parsing_model.predict(image, det_bbox)

    img_mask_ = np.isin(parse_maps, [4, 5, 6, 7])
    parse_maps[~img_mask_] = 0

    output_img = Image.fromarray(np.asarray(parse_maps, dtype=np.uint8))
    palette = get_palette(18)
    output_img.putpalette(palette)
    output_img.save(f"results/{os.path.basename(img_path)}-parsing.png")
    pdb.set_trace()


def test_hq_sam_model():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import cv2
    from segment_anything_hq import sam_model_registry, SamPredictor
    import os

    sam_checkpoint = "./checkpoints/preprocess/sam_hq_vit_tiny.pth"
    model_type = "vit_tiny"

    weight_dtype = torch.float32

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device, dtype=weight_dtype)
    sam.eval()
    predictor = SamPredictor(sam)
    img_path = "assets/app_examples/models/img_5.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    h, w = image.shape[:2]

    input_box = np.array([[0, 0, w, h]])
    input_point, input_label = np.array([[w//2, h//2]]), np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=False,
        hq_token_only=False,
    )
    mask = masks[0]
    image[mask] = 0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("results/test2.png", image)
    pdb.set_trace()


if __name__ == '__main__':
    # test_yolox_model()
    # test_pose_model()
    # test_parsing_model()
    test_hq_sam_model()
