#!/usr/bin/env python

# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import sys
import os
from pathlib import Path
from typing import List
from lxml import etree as ET
import mmcv
from mmdet.apis import (inference_detector, init_detector)
from tqdm import tqdm


def generate_class_colors(class_num):
    import cv2
    import numpy as np
    colors = 255 * np.ones((class_num, 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(0, 179, class_num)
    colors = cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR)[0]
    return colors


def draw_legand(img, origin, classes, colors, ssz: int = 16):
    import cv2
    c_num = len(classes)
    x, y = origin[0], origin[1]
    for c in range(c_num):
        color = colors[c]
        color = (int(color[0]), int(color[1]), int(color[2]))
        text = classes[c]
        img = cv2.rectangle(img, (x, y), (x + ssz - 1, y + ssz - 1), color, -1)
        img = cv2.putText(img, text, (x + ssz, y + ssz), cv2.FONT_HERSHEY_PLAIN,
                          1, (255, 0, 0), 1, cv2.LINE_AA)
        y += ssz
    return img


class LayoutDetector:
    def __init__(self, config: str, checkpoint: str, device: str):
        print(f'load from config={config}, checkpoint={checkpoint}')
        self.load(config, checkpoint, device)
        cfg = mmcv.Config.fromfile(config)
        self.classes = cfg.classes
        self.colors = generate_class_colors(len(self.classes))

    def load(self, config: str, checkpoint: str, device: str):
        self.model = init_detector(config, checkpoint, device)

    def predict(self, img_path: str):
        return inference_detector(self.model, img_path)

    def show(self, img, img_path: str, result, score_thr: float = 0.3, border: int = 3, show_legand: bool = True):
        import cv2
        if img is None:
            img = cv2.imread(img_path)

        if len(result) == 2:
            res_bbox = result[0]
        else:
            res_bbox = result

        for c in range(len(res_bbox)):
            color = self.colors[c]
            color = (int(color[0]), int(color[1]), int(color[2]))
            for pred in res_bbox[c]:
                if float(pred[4]) < score_thr:
                    continue
                x0, y0 = int(pred[0]), int(pred[1])
                x1, y1 = int(pred[2]), int(pred[3])
                img = cv2.rectangle(img, (x0, y0), (x1, y1), color, border)

        sz = max(img.shape[0], img.shape[1])
        scale = 1024.0 / sz
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

        if show_legand:
            ssz = 16
            c_num = len(self.classes)
            org_width = img.shape[1]
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, 8 * c_num, cv2.BORDER_REPLICATE)
            x = org_width
            y = img.shape[0] - ssz * c_num
            img = draw_legand(img, (x, y),
                              self.classes, self.colors, ssz=ssz)

        return img

    def draw_rects_with_data(self, img, result, score_thr: float = 0.3, border: int = 3, show_legand: bool = True):
        import cv2
        for c in range(len(result)):
            color = self.colors[c]
            color = (int(color[0]), int(color[1]), int(color[2]))
            for pred in result[c]:
                if float(pred[4]) < score_thr:
                    continue
                x0, y0 = int(pred[0]), int(pred[1])
                x1, y1 = int(pred[2]), int(pred[3])
                img = cv2.rectangle(img, (x0, y0), (x1, y1), color, border)

        sz = max(img.shape[0], img.shape[1])
        scale = 1024.0 / sz
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

        if show_legand:
            ssz = 16
            c_num = len(self.classes)
            org_width = img.shape[1]
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, 8 * c_num, cv2.BORDER_REPLICATE)
            x = org_width
            y = img.shape[0] - ssz * c_num
            img = draw_legand(img, (x, y), self.classes, self.colors, ssz=ssz)

        return img


def textblock_to_polygon(classes, res_segm, min_bbox_size=5):
    import cv2
    import numpy as np

    tb_cls_id = classes.index('text_block')
    polygons = []

    for segm in res_segm[tb_cls_id]:
        mask_img = segm.astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 輪郭、階層の抽出
        if len(contours)==0: # 領域が存在しないケース
            polygons.append(None)
            continue

        # 領域が存在するケース
        # 複数のcontourに分裂している場合、頂点数の多いもののみを主領域として採用する。 # method4とする
        main_contours_id = 0
        for i in range(len(contours)):
            if len(contours[i]) > len(contours[main_contours_id]):
                main_contours_id = i
        if len(contours[main_contours_id])<4:
            # 主領域が小さい場合、領域は存在しないものとして除外
            polygons.append(None)
            continue
        arclen = cv2.arcLength(contours[main_contours_id], True)
        app_cnt = cv2.approxPolyDP(contours[main_contours_id], epsilon=0.001 * arclen, closed=True)
        _, _, w, h = make_bbox_from_poly(app_cnt)
        if w < min_bbox_size and h < min_bbox_size:
            continue
        polygons.append(app_cnt)

    return polygons


def add_text_block_head(s, poly, conf=0.0, indent=''):
    s += indent + f'<TEXTBLOCK CONF = "{conf:0.3f}">\n'
    s += indent + '  ' + '<SHAPE><POLYGON POINTS = "'
    for id_poly, pt in enumerate(poly):
        if id_poly > 0:
            s += f',{pt[0][0]},{pt[0][1]}'
        else:
            s += f'{pt[0][0]},{pt[0][1]}'
    s += '"/></SHAPE>\n'
    return s


def add_block_ad_head(s, block_ad, conf=0.0):
    # block_ad [x1, y1, x2, y2, score]
    x, y = int(block_ad[0]), int(block_ad[1])
    w, h = int(block_ad[2] - block_ad[0]), int(block_ad[3] - block_ad[1])
    s += f'<BLOCK TYPE="広告" X="{x}" Y="{y}" WIDTH="{w}" HEIGHT="{h}" CONF = "{conf:0.3f}">\n'
    return s


def make_bbox_from_poly(poly):
    x1, y1 = poly[0][0][0], poly[0][0][1]
    x2, y2 = poly[0][0][0], poly[0][0][1]
    for pt in poly[1:]:
        x1 = min(x1, pt[0][0])
        y1 = min(y1, pt[0][1])
        x2 = max(x2, pt[0][0])
        y2 = max(y2, pt[0][1])

    return x1, y1, (x2-x1), (y2-y1)


def is_in_block_ad(block_ad, poly):
    if str(type(poly[0])) == "<class 'numpy.ndarray'>":
        x1, y1 = poly[0][0][0], poly[0][0][1]
        x2, y2 = poly[0][0][0], poly[0][0][1]
        for pt in poly[1:]:
            x1 = min(x1, pt[0][0])
            y1 = min(y1, pt[0][1])
            x2 = max(x2, pt[0][0])
            y2 = max(y2, pt[0][1])
    else:  # when poly is bbox
        x1, y1, x2, y2 = int(poly[0]), int(poly[1]), int(poly[2]), int(poly[3])
    cx = (x1+x2)//2
    cy = (y1+y2)//2

    # block_ad [x1, y1, x2, y2, score]
    if block_ad[0] < cx and cx < block_ad[2] and block_ad[1] < cy and cy < block_ad[3]:
        return True
    else:
        return False


def set_elm_detail(elm, bbox):
    elm.set('X', str(int(bbox[0])))
    elm.set('Y', str(int(bbox[1])))
    elm.set('WIDTH', str(int(bbox[2]-bbox[0])))
    elm.set('HEIGHT', str(int(bbox[3]-bbox[1])))
    elm.set('CONF', f'{bbox[4]:0.3f}')
    return

# Remove overlappiong polygons
def refine_tb_polygons(polygons, margin: int = 50):
    import cv2
    from copy import deepcopy
    res_polygons = deepcopy(polygons)

    for i, child_poly in enumerate(res_polygons):
        if child_poly is None:
            continue
        for j, parent_poly in enumerate(res_polygons):
            if i==j: # The child and parent are the same polygon
                continue
            if parent_poly is None:
                continue
            all_points_is_in = True
            for p in child_poly:
                x = int(p[0][0])
                y = int(p[0][1])
                if cv2.pointPolygonTest(parent_poly, (x, y), True) < -margin: # > 0 means in
                    all_points_is_in = False

            if  all_points_is_in:
                res_polygons[i] = None
                break

    return res_polygons

def get_relationship(res_bbox, tb_polygons, classes, use_block_ad: bool = True, score_thr: float = 0.3):
    import cv2
    tb_cls_id = classes.index('text_block')
    tb_info = [[] for i in range(len(tb_polygons))]
    independ_lines = []

    ad_info = None
    if use_block_ad:
        ba_cls_id = classes.index('block_ad')
        ad_info = [[] for i in range(len(res_bbox[ba_cls_id]))]

    if use_block_ad:
        for i, poly in enumerate(tb_polygons):
            if res_bbox[tb_cls_id][i][4] < score_thr or tb_polygons[i] is None:
                tb_info[i] = None
                continue
            for j, block_ad in enumerate(res_bbox[ba_cls_id]):
                if res_bbox[ba_cls_id][j][4] < score_thr:
                    ad_info[j] = None
                    continue
                if is_in_block_ad(block_ad, poly):
                    ad_info[j].append([tb_cls_id, i])
                    break

    for c in range(len(classes)):
        cls = classes[c]
        if not cls.startswith('line_'):
            continue
        for j, line in enumerate(res_bbox[c]):
            if float(line[4]) < score_thr:
                continue
            in_any_block = False
            # elems belonging to text_block
            for i, poly in enumerate(tb_polygons):
                if res_bbox[tb_cls_id][i][4] < score_thr or tb_polygons[i] is None:
                    tb_info[i] = None
                    continue
                cx, cy = (line[0]+line[2])//2, (line[1]+line[3])//2
                if cv2.pointPolygonTest(poly, (cx, cy), False) > 0:
                    tb_info[i].append([c, j])
                    in_any_block = True
                    break
            # elems belonging to ad_block
            if not in_any_block:
                for i, block_ad in enumerate(res_bbox[ba_cls_id]):
                    if ad_info[i] is None:
                        continue
                    if is_in_block_ad(block_ad, line):
                        ad_info[i].append([c, j])
                        in_any_block = True
                        break
            # Line elements not belonging to any text_block or ad_block
            if not in_any_block:
                independ_lines.append([c, j])

    return tb_info, ad_info, independ_lines


def refine_tb_relationship(tb_polygons, tb_info, classes, margin: int = 50):
    import cv2
    tb_cls_id = classes.index('text_block')

    for c_index, child_poly in enumerate(tb_polygons):
        if child_poly is None or tb_info[c_index] is None:
            continue
        for p_index, parent_poly in enumerate(tb_polygons):
            if c_index == p_index:  # The child and parent are the same polygon
                continue
            if parent_poly is None or tb_info[p_index] is None:
                continue
            all_points_is_in = True
            for p in child_poly:
                x = int(p[0][0])
                y = int(p[0][1])
                if cv2.pointPolygonTest(parent_poly, (x, y), True) < -margin:
                    # cv2.pointPolygonTest () > 0 means (x,y) is in parent_poly
                    all_points_is_in = False

            if all_points_is_in:  # c is in p
                if len(tb_info[c_index]) == 0:
                    tb_info[p_index].append([tb_cls_id, c_index])
                    tb_info[c_index] = None
                else:  # tb[i] has childen
                    for child_elm in tb_info[c_index]:
                        tb_info[p_index].append(child_elm)
                    tb_info[c_index] = None
                break

    # merge text blocks
    for i in range(len(tb_info)):
        have_only_tb = True
        if tb_info[i] is None:
            continue
        for c_id, _ in tb_info[i]:
            if c_id != tb_cls_id:
                have_only_tb = False
                break
        if have_only_tb:
            tb_info[i] = []

    return tb_info


def convert_to_xml_string2(img_w, img_h, img_path, classes, result,
                           score_thr: float = 0.3,
                           min_bbox_size: str = 5,
                           use_block_ad: bool = True):
    import cv2
    from .ndl_parser import name_to_org_name

    img_name = os.path.basename(img_path)
    s = f'<PAGE IMAGENAME = "{img_name}" WIDTH = "{img_w}" HEIGHT = "{img_h}">\n'

    res_bbox = result[0]
    res_segm = result[1]

    # convert text block masks to polygons
    tb_polygons = textblock_to_polygon(classes, res_segm, min_bbox_size)
    tb_info, ad_info, independ_lines = get_relationship(res_bbox, tb_polygons, classes)
    # refine text blocks : remove overlapping text blocks
    tb_info = refine_tb_relationship(tb_polygons, tb_info, classes, margin=50)

    tb_cls_id = classes.index('text_block')
    if use_block_ad:
        ba_cls_id = classes.index('block_ad')
        for i_ba, block_ad in enumerate(res_bbox[ba_cls_id]):
            if ad_info[i_ba] is None:
                continue
            s += '  '
            s = add_block_ad_head(s, block_ad, block_ad[4])
            for c, j in ad_info[i_ba]:
                if c == tb_cls_id:
                    if tb_info[j] is None:
                        continue
                    s = add_text_block_head(s, tb_polygons[j], res_bbox[tb_cls_id][j][4], '    ')
                    # add lines in textblock in block_ad
                    if len(tb_info[j]) == 0:
                        # create and add a line_main elem at least one
                        x, y, w, h = make_bbox_from_poly(tb_polygons[j])
                        if w >= min_bbox_size and h >= min_bbox_size:
                            s += f'      <LINE TYPE = "{name_to_org_name(classes[0])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}"></LINE>\n'
                    else:
                        for c_id, i in tb_info[j]:
                            line = res_bbox[c_id][i]
                            conf = float(line[4])
                            if conf < score_thr:
                                continue
                            if c_id == tb_cls_id:  # write as Line_main
                                x, y, w, h = make_bbox_from_poly(tb_polygons[i])
                                if w >= min_bbox_size and h >= min_bbox_size:
                                    s += f'      <LINE TYPE = "{name_to_org_name(classes[0])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}"></LINE>\n'
                            else:
                                x, y = int(line[0]), int(line[1])
                                w, h = int(line[2] - line[0]), int(line[3] - line[1])
                                s += f'      <LINE TYPE = "{name_to_org_name(classes[c_id])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
                    s += '    </TEXTBLOCK>\n'
                    tb_info[j] = None
                else:
                    line = res_bbox[c][j]
                    conf = float(line[4])
                    if conf < score_thr:
                        continue
                    x, y = int(line[0]), int(line[1])
                    w, h = int(line[2] - line[0]), int(line[3] - line[1])
                    s += f'    <LINE TYPE = "{name_to_org_name(classes[c])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
            s += '    </BLOCK>\n'

    # Text block and line elems inside the text block
    for j in range(len(tb_info)):
        if tb_info[j] is None or res_bbox[tb_cls_id][j][4] < score_thr or tb_polygons[j] is None:  # text block already converted
            continue
        s = add_text_block_head(s, tb_polygons[j], res_bbox[tb_cls_id][j][4], '  ')
        if len(tb_info[j]) == 0:  # text block without line elms
            # create and add a line_main elem at least one
            x, y, w, h = make_bbox_from_poly(tb_polygons[j])
            if w >= min_bbox_size and h >= min_bbox_size:
                s += f'    <LINE TYPE = "{name_to_org_name(classes[0])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}"></LINE>\n'
        else:
            for c_id, i in tb_info[j]:
                line = res_bbox[c_id][i]
                conf = float(line[4])
                if conf < score_thr:
                    continue
                if c_id == tb_cls_id:  # write as line_main
                    x, y, w, h = make_bbox_from_poly(tb_polygons[i])
                    if w >= min_bbox_size and h >= min_bbox_size:
                        s += f'    <LINE TYPE = "{name_to_org_name(classes[0])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}"></LINE>\n'
                else:
                    x, y = int(line[0]), int(line[1])
                    w, h = int(line[2] - line[0]), int(line[3] - line[1])
                    s += f'    <LINE TYPE = "{name_to_org_name(classes[c_id])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
        s += '  </TEXTBLOCK>\n'

    # Line elems outside text_block and block_ad
    for c, j in independ_lines:
        line = res_bbox[c][j]
        conf = float(line[4])
        if conf < score_thr:
            continue
        x, y = int(line[0]), int(line[1])
        w, h = int(line[2] - line[0]), int(line[3] - line[1])
        s += f'  <LINE TYPE = "{name_to_org_name(classes[c])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'

    # Block elms other than block_ad
    for c in range(len(classes)):
        cls = classes[c]
        if c != ba_cls_id and cls.startswith('block_'):
            for block in res_bbox[c]:
                conf = float(block[4])
                if conf < score_thr:
                    continue
                x, y = int(block[0]), int(block[1])
                w, h = int(block[2] - block[0]), int(block[3] - block[1])
                s += f'  <BLOCK TYPE = "{name_to_org_name(cls)}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></BLOCK>\n'

    s += '</PAGE>\n'

    return s

def convert_to_xml_string_with_data(img_w, img_h, img_path, classes, result,
                                    score_thr: float = 0.3, min_bbox_size: str = 5, use_block_ad: bool = True):
    import cv2
    import copy
    import numpy as np
    import time
    from .ndl_parser import name_to_org_name

    img_name = os.path.basename(img_path)
    s = f'<PAGE IMAGENAME = "{img_name}" WIDTH = "{img_w}" HEIGHT = "{img_h}">\n'

    res_bbox = result[0]
    res_segm = result[1]

    tb_polygons = textblock_to_polygon(classes, res_segm, min_bbox_size)
    tb_polygons = refine_tb_polygons(tb_polygons)
    tb_info, ad_info, independ_lines =  get_relationship(res_bbox, tb_polygons, classes)

    tb_cls_id = classes.index('text_block')
    if use_block_ad:
        ba_cls_id = classes.index('block_ad')
        for i_ba, block_ad in enumerate(res_bbox[ba_cls_id]):
            if ad_info[i_ba] is None:
                continue
            s += '  '
            s = add_block_ad_head(s, block_ad, block_ad[4])
            for c, j in ad_info[i_ba]:
                if c == tb_cls_id:
                    s = add_text_block_head(s, tb_polygons[j], res_bbox[tb_cls_id][j][4], '    ')
                    # add lines in textblock in block_ad
                    for c_id, i in tb_info[j]:
                        line = res_bbox[c_id][i]
                        conf = float(line[4])
                        if conf < score_thr:
                            continue
                        x, y = int(line[0]), int(line[1])
                        w, h = int(line[2] - line[0]), int(line[3] - line[1])
                        s += f'      <LINE TYPE = "{name_to_org_name(classes[c_id])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
                    s += '    </TEXTBLOCK>\n'
                    tb_info[j] = None
                else:
                    line = res_bbox[c][j]
                    conf = float(line[4])
                    if conf < score_thr:
                        continue
                    x, y = int(line[0]), int(line[1])
                    w, h = int(line[2] - line[0]), int(line[3] - line[1])
                    s += f'    <LINE TYPE = "{name_to_org_name(classes[c])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
            s += f'    </BLOCK>\n'

    # Text block and line elems inside the text block
    for j in range(len(tb_info)):
        if tb_info[j] is None or res_bbox[tb_cls_id][j][4] < score_thr or tb_polygons[j] is None: # text block already converted
            continue
        s = add_text_block_head(s, tb_polygons[j], res_bbox[tb_cls_id][j][4], '  ')
        if len(tb_info[j])==0: # text block without line elms
            # create and add a line_main elem at least one
            x, y, w, h = make_bbox_from_poly(tb_polygons[j])
            if w >= min_bbox_size and h >= min_bbox_size:
                s += f'    <LINE TYPE = "{name_to_org_name(classes[0])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}"></LINE>\n'
        else:
            for c_id, i in tb_info[j]:
                line = res_bbox[c_id][i]
                conf = float(line[4])
                if conf < score_thr:
                    continue
                x, y = int(line[0]), int(line[1])
                w, h = int(line[2] - line[0]), int(line[3] - line[1])
                s += f'    <LINE TYPE = "{name_to_org_name(classes[c_id])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
        s += '  </TEXTBLOCK>\n'

    # Line elems outside text_block and block_ad
    for c, j in independ_lines:
        line = res_bbox[c][j]
        conf = float(line[4])
        # if conf < score_thr:
        #     continue
        x, y = int(line[0]), int(line[1])
        w, h = int(line[2] - line[0]), int(line[3] - line[1])
        s += f'  <LINE TYPE = "{name_to_org_name(classes[c])}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'

    # Block elms other than block_ad
    for c in range(len(classes)):
        cls = classes[c]
        if c != ba_cls_id and cls.startswith('block_'):
            for block in res_bbox[c]:
                conf = float(block[4])
                if conf < score_thr:
                    continue
                x, y = int(block[0]), int(block[1])
                w, h = int(block[2] - block[0]), int(block[3] - block[1])
                s += f'  <BLOCK TYPE = "{name_to_org_name(cls)}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></BLOCK>\n'

    s += '</PAGE>\n'

    return s

def run_layout_detection(img_paths: List[str] = None, list_path: str = None, output_path: str = "layout_prediction.xml",
                         config: str = './models/config_file.py',
                         checkpoint: str = 'models/weight_file.pth',
                         device: str = 'cuda:0', score_thr: float = 0.3,
                         use_show: bool = False, dump_dir: str = None,
                         use_time: bool = False):
    if use_time:
        import time
        st_time = time.time()

    if list_path is not None:
        img_paths = list([s.strip() for s in open(list_path).readlines()])
    if img_paths is None:
        print('Please specify --img_paths or --list_path')
        return -1

    detector = LayoutDetector(config, checkpoint, device)

    if dump_dir is not None:
        Path(dump_dir).mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        def tee(s):
            print(s, file=f, end="")
            print(s, file=sys.stdout, end="")

        tee('<?xml version="1.0" encoding="utf-8" standalone="yes"?><OCRDATASET xmlns="">\n')
        if use_time:
            head_time = time.time()

        for img_path in tqdm(img_paths):
            import cv2
            img = cv2.imread(img_path)

            result = detector.predict(img)

            if use_show:
                dump_img = detector.show(img, img_path, result, score_thr=score_thr)
                cv2.namedWindow('show')
                cv2.imshow('show', dump_img)
                if 27 == cv2.waitKey(0):
                    break

            if dump_dir is not None:
                import cv2
                dump_img = detector.show(img, img_path, result, score_thr=score_thr)
                cv2.imwrite(str(Path(dump_dir) / Path(img_path).name), dump_img)
            img_h, img_w = img.shape[0:2]
            xml_str = convert_to_xml_string2(
                img_w, img_h, img_path, detector.classes, result, score_thr=score_thr)
            tee(xml_str)

        tee('</OCRDATASET>\n')
    # end with
    if use_time:
        end_time = time.time()
        print("+---------------------------------------+")
        print(f"all elapsed time        : {end_time-st_time:0.6f} [sec]")
        print(f"head time               : {head_time-st_time:0.6f} [sec]")
        infer_time = end_time-head_time
        infer_per_img = infer_time / len(img_paths)
        print(f"inference & xmlize time : {infer_time:0.6f} [sec]")
        print(f"                per img : {infer_per_img:0.6f} [sec]")
        print("+---------------------------------------+")


class InferencerWithCLI:
    def __init__(self, conf_dict):
        config = conf_dict['config_path']
        checkpoint = conf_dict['checkpoint_path']
        device = conf_dict['device']
        self.detector = LayoutDetector(config, checkpoint, device)

    def inference_with_cli(self, img, img_path,
                           score_thr: float = 0.3, dump: bool = False):

        node = ET.fromstring(
            '<?xml version="1.0" standalone="yes"?><OCRDATASET xmlns="">\n</OCRDATASET>\n')

        # prediction
        if self.detector is None:
            print('ERROR: Layout detector is not created.')
            return None
        result = self.detector.predict(img)

        # xml creation
        xml_str = convert_to_xml_string_with_data(
            img.shape[1], img.shape[0], img_path, self.detector.classes, result, score_thr=score_thr)

        # xml conversion from string
        result_xml = ET.fromstring(xml_str)
        node.append(result_xml)
        tree = ET.ElementTree(node)

        # xml conversion from string
        dump_img = None

        return {'xml': tree, 'dump_img': dump_img}


if __name__ == '__main__':
    from .utils import auto_run
    auto_run(run_layout_detection)
