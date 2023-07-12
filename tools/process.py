#!/usr/bin/env python

# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import sys
import os
from pathlib import Path
from .utils import auto_run
from typing import List
import xml.etree.ElementTree as ET
import mmengine
import mmcv
from mmdet.apis import (inference_detector, init_detector)


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
        cfg = mmengine.Config.fromfile(config)
        self.classes = cfg.classes
        self.colors = generate_class_colors(len(self.classes))

    def load(self, config: str, checkpoint: str, device: str):
        self.model = init_detector(config, checkpoint, device)

    def predict(self, img_path: str):
        return inference_detector(self.model, img_path)

    def show(self, img_path: str, result, score_thr: float = 0.3, border: int = 3, show_legand: bool = True):
        import cv2
        img = cv2.imread(img_path)
        for pred,score,c in zip(result.pred_instances.bboxes,result.pred_instances.scores,result.pred_instances.labels):
            color = self.colors[c]
            color = (int(color[0]), int(color[1]), int(color[2]))
            if float(score) < score_thr:
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
        for pred,score,c in zip(result.pred_instances.bboxes,result.pred_instances.scores,result.pred_instances.labels):
            color = self.colors[c]
            color = (int(color[0]), int(color[1]), int(color[2]))
            if float(score) < score_thr:
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


def convert_to_xml_string(img_path, classes, result, score_thr: float = 0.3):
    import cv2
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[0:2]

    from .ndl_parser import name_to_org_name

    img_name = os.path.basename(img_path)
    s = f'<PAGE IMAGENAME = "{img_name}" WIDTH = "{img_w}" HEIGHT = "{img_h}">\n'
    for pred,score,c in zip(result.pred_instances.bboxes,result.pred_instances.scores,result.pred_instances.labels):
        cls = classes[c]
        if cls.startswith('line_'):
            line = pred
            conf = float(score])
            if conf < score_thr:
                continue
            x, y = int(line[0]), int(line[1])
            w, h = int(line[2] - line[0]), int(line[3] - line[1])
            s += f'<LINE TYPE = "{name_to_org_name(cls)}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
        elif cls.startswith('block_'):
            block = pred
            conf = float(score)
            if conf < score_thr:
                continue
            x, y = int(block[0]), int(block[1])
            w, h = int(block[2] - block[0]), int(block[3] - block[1])
            s += f'<BLOCK TYPE = "{name_to_org_name(cls)}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></BLOCK>\n'

    s += '</PAGE>\n'
    return s


def convert_to_xml_string_with_data(img, img_path, classes, result, score_thr: float = 0.3):
    img_h, img_w = img.shape[0:2]

    from .ndl_parser import name_to_org_name

    img_name = os.path.basename(img_path)
    s = f'<PAGE IMAGENAME = "{img_name}" WIDTH = "{img_w}" HEIGHT = "{img_h}">\n'

    for pred,score,c in zip(result.pred_instances.bboxes,result.pred_instances.scores,result.pred_instances.labels):
        cls = classes[c]
        if cls.startswith('line_'):
            line = pred
            conf = float(score)
            if conf < score_thr:
                continue
            x, y = int(line[0]), int(line[1])
            w, h = int(line[2] - line[0]), int(line[3] - line[1])
            s += f'<LINE TYPE = "{name_to_org_name(cls)}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></LINE>\n'
        elif cls.startswith('block_'):
            block = pred
            conf = float(score)
            if conf < score_thr:
                continue
            x, y = int(block[0]), int(block[1])
            w, h = int(block[2] - block[0]), int(block[3] - block[1])
            s += f'<BLOCK TYPE = "{name_to_org_name(cls)}" X = "{x}" Y = "{y}" WIDTH = "{w}" HEIGHT = "{h}" CONF = "{conf:0.3f}"></BLOCK>\n'

    s += '</PAGE>\n'
    return s


def run_layout_detection(img_paths: List[str] = None, list_path: str = None, output_path: str = "layout_prediction.xml",
                         config: str = './models/20210604_from_competition_separated_deskewed/cascade_rcnn_r50_fpn_1x_ndl_1024.py',
                         checkpoint: str = 'models/20210604_from_competition_separated_deskewed/latest.pth',
                         device: str = 'cuda:0', score_thr: float = 0.3, use_show: bool = False, dump_dir: str = None):
    detector = LayoutDetector(config, checkpoint, device)
    if list_path is not None:
        img_paths = list([s.strip() for s in open(list_path).readlines()])
    if img_paths is None:
        print('Please specify --img_paths or --list_path')
        return -1
    if dump_dir is not None:
        Path(dump_dir).mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        def tee(s):
            print(s, file=f, end="")
            print(s, file=sys.stdout, end="")

        tee('<?xml version="1.0" encoding="utf-8" standalone="yes"?><OCRDATASET xmlns="">\n')

        for img_path in img_paths:
            result = detector.predict(img_path)
            xml_str = convert_to_xml_string(
                img_path, detector.classes, result, score_thr=score_thr)
            tee(xml_str)

            if use_show:
                import cv2
                img = detector.show(img_path, result, score_thr=score_thr)
                cv2.namedWindow('show')
                cv2.imshow('show', img)
                if 27 == cv2.waitKey(0):
                    break

            if dump_dir is not None:
                import cv2
                img = detector.show(img_path, result, score_thr=score_thr)
                cv2.imwrite(str(Path(dump_dir) / Path(img_path).name), img)

        tee('</OCRDATASET>\n')


class InferencerWithCLI:
    def __init__(self, conf_dict):
        config = conf_dict['config_path']
        checkpoint = conf_dict['checkpoint_path']
        device = conf_dict['device']
        self.detector = LayoutDetector(config, checkpoint, device)

    def inference_wich_cli(self, img=None, img_path='',
                           score_thr: float = 0.3, dump: bool = False):
        ET.register_namespace('', 'NDLOCRDATASET')
        node = ET.fromstring(
            '<?xml version="1.0" encoding="utf-8" standalone="yes"?><OCRDATASET xmlns="">\n</OCRDATASET>\n')

        # prediction
        if self.detector is None:
            print('ERROR: Layout detector is not created.')
            return None
        result = self.detector.predict(img)

        # xml creation
        xml_str = convert_to_xml_string_with_data(
            img, img_path, self.detector.classes, result, score_thr=score_thr)

        # xml conversion from string
        result_xml = ET.fromstring(xml_str)
        node.append(result_xml)
        tree = ET.ElementTree(node)

        # xml conversion from string
        dump_img = None
        if dump is not None:
            dump_img = self.detector.draw_rects_with_data(
                img, result, score_thr=score_thr)

        return {'xml': tree, 'dump_img': dump_img}


if __name__ == '__main__':
    auto_run(run_layout_detection)
