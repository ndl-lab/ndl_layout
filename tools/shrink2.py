#!/usr/bin/env python

# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from pathlib import Path
from tqdm import tqdm
from .utils import auto_run
import cv2
import os


def main(src_dir: str, dst_dir: str, prefix: str = ""):
    os.makedirs(dst_dir, exist_ok=True)

    for path in tqdm(list(Path(src_dir).iterdir())):
        if path.is_file() or path.is_symlink():
            path = str(path)
            img = cv2.imread(path)
            if '.jpg' in path:
                img_ = cv2.resize(img, None, fx=0.5, fy=0.5,
                                  interpolation=cv2.INTER_AREA)
            else:
                img_ = cv2.resize(img, None, fx=0.5, fy=0.5,
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(Path(dst_dir) / (prefix + Path(path).name)), img_)


auto_run(main)
