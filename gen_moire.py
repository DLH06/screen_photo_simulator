import random
import cv2
import numpy as np
import os, glob
import pandas as pd
from tqdm import tqdm

from src.moire import linear_wave, dither, nonlinear_wave
from src.module import RecaptureModule


def generate(
    canvas: str, 
    recapture_verbose: bool = False, 
    gamma: float = 1.0, 
    show_mask: bool = False, 
    seed: int = 42
):
    original = canvas.copy()
    H, W, _ = canvas.shape
    dst_H, dst_W, _ = original.shape

    src_pt = np.zeros((4,2), dtype="float32")
    src_pt[0] = [W // 4, H // 4]
    src_pt[1] = [W // 4 * 3, H // 4]
    src_pt[2] = [W // 4 * 3, H // 4 * 3]
    src_pt[3] = [W // 4, H // 4 * 3]

    dst_pt = np.zeros((4,2), dtype="float32")
    dst_pt[0] = [dst_W // 4, dst_H // 4]    # top-left
    dst_pt[1] = [dst_W // 4 * 3, dst_H // 4]   # top-right
    dst_pt[2] = [dst_W // 4 * 3, dst_H // 4 * 3]   # bottom-right
    dst_pt[3] = [dst_W // 4, dst_H // 4 * 3]   # bottom-left

    recap_module = RecaptureModule(dst_H, dst_W,
                                   v_moire=0, v_type='sg', v_skew=[20, 80], v_cont=10, v_dev=3,
                                   h_moire=0, h_type='f', h_skew=[20, 80], h_cont=10, h_dev=3,
                                   nl_moire=True, nl_dir='b', nl_type='sine', nl_skew=0,
                                   nl_cont=10, nl_dev=3, nl_tb=0.15, nl_lr=0.15,
                                   gamma=gamma, margins=None, seed=seed)
    try:
        canvas, _ = recap_module(canvas,
                          new_src_pt = src_pt,
                          verbose=recapture_verbose,
                          show_mask=show_mask)
    except Exception:
        canvas = recap_module(canvas,
                          new_src_pt = src_pt,
                          verbose=recapture_verbose,
                          show_mask=show_mask)

    r, g, b = random.randint(0, 224), random.randint(0, 224), random.randint(0, 224)
    if random.random() < 0.33:
        canvas = dither(
            canvas,
            rowwise=True,
            color=(r,g,b),
            gap=random.randint(1, 20), 
            skew=random.randint(1, 100), 
            contrast=random.randint(1, 50),
            pattern=random.choice(['rgb', 'single']), 
        )
    elif random.random() < 0.66:
        canvas = linear_wave(
            canvas,
            gap=random.randint(1, 20),
            skew=0,
            rowwise=True,
            color=(r,g,b),
            thick=random.randint(1, 4),
            contrast=random.randint(1, 50),
            dev=random.randint(1, 5),
            pattern=random.choice(['fixed', 'sine', 'gaussian']),
            seed=random.choice([True, False]),
        )
    else:
        canvas, _ = nonlinear_wave(
            canvas, 
            skew=0,
            color=(r,g,b),
            gap=random.randint(1, 4),
            thick=random.randint(1, 3),
            contrast=random.randint(1, 20),
            dev=1,
            seed=random.randint(1, 100),
            directions=random.choice(['b', 'h', 'v']),
            pattern=random.choice(['sine', 'fixed', 'gaussian']),
            tb_margins=0.3,
            lr_margins=0.3,
        )

    return canvas
