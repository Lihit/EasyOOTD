# -*- coding: utf-8 -*-

import numpy as np


def align_pose(kpts2d, kpts2d_ref):
    kpt_thred = 0.4
    kpts2d_align = kpts2d.copy()
    # 先做整体的缩放
    org_top_h = kpts2d[[5, 6], 1].min()
    org_down_h = kpts2d[[11, 12], 1].max()
    org_height = org_down_h - org_top_h

    ref_top_h = kpts2d_ref[[5, 6], 1].min()
    ref_down_h = kpts2d_ref[[11, 12], 1].max()
    ref_height = ref_down_h - ref_top_h

    h_scale = ref_height / org_height
    print("scale height:", h_scale)
    kpts2d_align[:, 1] *= h_scale

    org_left_h = kpts2d[[6, 12], 0].min()
    org_right_h = kpts2d[[5, 11], 0].max()
    org_width = org_right_h - org_left_h

    ref_left_h = kpts2d_ref[[6, 12], 0].min()
    ref_right_h = kpts2d_ref[[5, 11], 0].max()
    ref_width = ref_right_h - ref_left_h

    w_scale = ref_width / org_width
    print("scale width:", w_scale)
    kpts2d_align[:, 0] *= w_scale

    ref_mid = (kpts2d_ref[11, :2] + kpts2d_ref[12, :2]) / 2.0
    align_mid = (kpts2d_align[11, :2] + kpts2d_align[12, :2]) / 2.0

    kpts2d_align[:, :2] = kpts2d_align[:, :2] - align_mid[None] + ref_mid[None]
    return kpts2d_align
