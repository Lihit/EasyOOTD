
import pdb

import onnxruntime as ort
import onnx
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _box2cs(box, input_size):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w, h], dtype=np.float32)

    scale = scale * 1.0

    return center, scale


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt


class HumanParsingModel:
    """
    Human Parsing Model
    """

    def __init__(self, **kwargs):
        model_path = kwargs.get("model_path", "")

        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']

        print(f"OnnxRuntime use {providers}")
        opts = ort.SessionOptions()
        # opts.inter_op_num_threads = kwargs.get("num_threads", 4)
        # opts.intra_op_num_threads = kwargs.get("num_threads", 4)
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)
        self.mean = np.array([0.406, 0.456, 0.485]).reshape(1, 1, 3)
        self.std = np.array([0.225, 0.224, 0.229]).reshape(1, 1, 3)

    def preprocess(self, *data):
        img, bbox = data
        h, w = img.shape[:2]
        INPUT_H, INPUT_W = self.session.get_inputs()[0].shape[2:4]
        input_size = [INPUT_W, INPUT_H]
        c, s = _box2cs(bbox, input_size)
        r = 0
        trans = get_affine_transform(c, s, r, input_size)
        img = cv2.warpAffine(
            img,
            trans, (int(input_size[0]), int(input_size[1])),
            flags=cv2.INTER_LINEAR)
        img = img / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))[None]
        img_info = {"c": c, "r": r, "s": s, "input_size": input_size, "img_dim": [h, w]}
        return img.astype(np.float32), [img_info]

    def postprocess(self, *data):
        logits_preds, img_metas = data
        input_w, input_h = img_metas[0]["input_size"]
        logits_preds = F.interpolate(torch.from_numpy(logits_preds), size=(input_h, input_w), mode='bilinear',
                                     align_corners=True)
        logits_preds = np.argmax(logits_preds.cpu().numpy(), axis=1)
        parse_maps = []
        for i in range(logits_preds.shape[0]):
            img_meta = img_metas[i]
            trans_inv = get_affine_transform(img_meta["c"], img_meta["s"], 0, img_meta["input_size"], inv=True)
            img_h, img_w = img_meta["img_dim"]
            logits = logits_preds[i, :, :]
            parse_maps.append(cv2.warpAffine(
                logits,
                trans_inv,
                (img_w, img_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[0]
            ))

        return np.stack(parse_maps)

    def predict(self, *data):
        image, img_info = self.preprocess(*data)
        ort_inputs = {self.session.get_inputs()[0].name: image}
        preds = self.session.run(None, ort_inputs)
        preds = self.postprocess(preds[0], img_info)
        return preds[0]
