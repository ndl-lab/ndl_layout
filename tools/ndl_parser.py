#!/usr/bin/env python

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import List
from .utils import auto_run
from enum import IntEnum, auto


class Category(IntEnum):
    LINE_MAIN = 0
    LINE_INOTE = auto()
    LINE_HNOTE = auto()
    LINE_CAPTION = auto()
    BLOCK_FIG = auto()
    BLOCK_TABLE = auto()
    BLOCK_PILLAR = auto()
    BLOCK_FOLIO = auto()
    BLOCK_RUBI = auto()
    BLOCK_CHART = auto()
    BLOCK_EQN = auto()
    BLOCK_CFM = auto()
    BLOCK_ENG = auto()
    CHAR = auto()
    NUM = auto()

# TYPE=“本文|割注|頭注|キャプション"
# TYPE=“図版|表組|柱|ノンブル|ルビ|組織図|数式|化学式|欧文|


categories = [
    {'id': int(Category.LINE_MAIN),    'name': 'line_main',    'org_name': '本文'},
    {'id': int(Category.LINE_INOTE),   'name': 'line_inote',   'org_name': '割注'},
    {'id': int(Category.LINE_HNOTE),   'name': 'line_hnote',   'org_name': '頭注'},
    {'id': int(Category.LINE_CAPTION), 'name': 'line_caption', 'org_name': 'キャプション'},
    {'id': int(Category.BLOCK_FIG),    'name': 'block_fig',    'org_name': '図版'},
    {'id': int(Category.BLOCK_TABLE),  'name': 'block_table',  'org_name': '表組'},
    {'id': int(Category.BLOCK_PILLAR), 'name': 'block_pillar', 'org_name': '柱'},
    {'id': int(Category.BLOCK_FOLIO),  'name': 'block_folio',  'org_name': 'ノンブル'},
    {'id': int(Category.BLOCK_RUBI),   'name': 'block_rubi',   'org_name': 'ルビ'},
    {'id': int(Category.BLOCK_CHART),  'name': 'block_chart',  'org_name': '組織図'},
    {'id': int(Category.BLOCK_EQN),    'name': 'block_eqn',    'org_name': '数式'},
    {'id': int(Category.BLOCK_CFM),    'name': 'block_cfm',    'org_name': '化学式'},
    {'id': int(Category.BLOCK_ENG),    'name': 'block_eng',    'org_name': '欧文'},
    {'id': int(Category.CHAR),         'name': 'char',         'org_name': 'char'},
    {'id': int(Category.NUM),          'name': 'void',         'org_name': 'void'}]

categories_org_name_index = {elem['org_name']: elem for elem in categories}
categories_name_index = {elem['name']: elem for elem in categories}


def org_name_to_id(s: str):
    return categories_org_name_index[s]['id']


def name_to_org_name(s: str):
    return categories_name_index[s]['org_name']


class NDLObject:
    def __init__(self, x, y, width, height, category_id=-1):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.category_id = category_id

    def __repr__(self):
        return f'NDLObject({self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


class NDLBlock(NDLObject):
    def __init__(self, type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_id = org_name_to_id(type)
        self.type = type

    def __repr__(self):
        return f'NDLBlock({self.type}, {self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


class NDLChar(NDLObject):
    def __init__(self, moji: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moji = moji
        self.category_id = Category.CHAR

    def __repr__(self):
        return f'NDLChar(\'{self.moji}\', {self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


class NDLLine(NDLObject):
    def __init__(self, chars: List[NDLChar], opt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chars = chars
        self.category_id = org_name_to_id(opt)
        self.opt = opt

    def __repr__(self):
        return f'NDLLine({self.chars}, {self.opt}, {self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


class NDLPage:
    def __init__(self, img_path: str, objects: List[NDLObject], source_xml: str):
        self.img_path = img_path
        self.objects = objects
        self.source_xml = source_xml

    def __repr__(self):
        return f'NDLPage({self.img_path}, {self.objects}, {self.source_xml})'


class NDLDataset:
    def __init__(self, pages=None):
        self.pages = [] if pages is None else pages

    def parse(self, xml_path: str, img_dir: str):
        import xml.etree.ElementTree as ET
        from pathlib import Path

        print(f'loading from {xml_path} ... ', end='')

        tree = ET.parse(xml_path)
        root = tree.getroot()
        pages = []

        def parse_bbox(elem):
            return float(elem.attrib['X']), float(elem.attrib['Y']), float(elem.attrib['WIDTH']), float(elem.attrib['HEIGHT'])

        for page in root:
            img_path = str(Path(img_dir) / page.attrib['IMAGENAME'])
            objects = []
            for elem in page:
                bbox = parse_bbox(elem)
                prefix, has_namespace, postfix = elem.tag.partition('}')
                if has_namespace:
                    tag = postfix
                else:
                    tag = elem.tag
                if tag == 'BLOCK':
                    objects.append(NDLBlock(elem.attrib['TYPE'], *bbox))
                elif tag == 'LINE':
                    chars = []
                    for char in elem:
                        bbox_char = parse_bbox(char)
                        if char.get('MOJI') is None:
                            continue
                        chars.append(NDLChar(char.attrib['MOJI'], *bbox_char))
                    # Changed OPT to TYPE specification.
                    # objects.append(NDLLine(chars, elem.attrib.get('OPT', ''), *bbox))
                    objects.append(
                        NDLLine(chars, elem.attrib.get('TYPE', ''), *bbox))

                else:
                    pass
            pages.append(NDLPage(img_path, objects, Path(xml_path).stem))
        print(f'done! {len(pages)} loaded')
        self.pages.extend(pages)

    def summary(self, output_dir: str = "./generated/"):
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        sizes = []
        bbox_nums = []
        opts = defaultdict(int)
        types = defaultdict(int)
        for page in self.pages:
            cnt = 0
            for obj in page.objects:
                sizes.append(
                    np.array([obj.width, obj.height], dtype=np.float32))
                if isinstance(obj, NDLBlock):
                    types[obj.type] += 1
                cnt += 1
                if isinstance(obj, NDLLine):
                    cnt += len(obj.chars)
                    opts[obj.opt] += 1
            bbox_nums.append(cnt)

        print(opts)
        print(types)

        sizes = np.array(sizes)
        bbox_nums = np.array(bbox_nums)

        def savefig(data, file_name):
            plt.figure()
            plt.hist(data)
            plt.savefig(output_dir + file_name)

        savefig(sizes[:, 0], "hist_width.png")
        savefig(sizes[:, 1], "hist_height.png")
        savefig(sizes[:, 1] / sizes[:, 0], "hist_aspect.png")
        savefig(bbox_nums, "hist_bbox_num.png")

    def to_coco_fmt(self, fx=1.0, fy=1.0, add_char: bool = True, add_block: bool = True, add_prefix: bool = False, suffix: str = ".jpg"):
        import cv2
        from pathlib import Path
        from tqdm import tqdm
        from collections import defaultdict
        output = {'images': [], 'annotations': []}
        image_id = 0
        annotation_id = 0
        instance_num = defaultdict(int)

        print("start to_coco_fmt")

        def make_bbox(obj):
            x1, y1 = fx * obj.x, fy * obj.y
            width, height = fx * obj.width, fy * obj.height
            x2, y2 = x1 + width, y1 + height
            bbox = [x1, y1, width, height]
            area = width * height
            contour = [x1, y1, x2, y1, x2, y2, x1, y2]
            return bbox, contour, area

        def add_annotation(obj):
            bbox, contour, area = make_bbox(obj)
            ann = {'image_id': image_id, 'id': annotation_id, 'bbox': bbox, 'area': area,
                   'iscrowd': 0, 'category_id': int(obj.category_id)}
            ann['segmentation'] = [contour]
            output['annotations'].append(ann)

        def add_line_annotation(obj):
            bbox, _, area_sum = make_bbox(obj)
            area = 0
            contours = []
            for char in obj.chars:
                _, contour, area_ = make_bbox(char)
                area += area_
                contours.append(contour)
            if area == 0:
                area = area_sum
            ann = {'image_id': image_id, 'id': annotation_id, 'bbox': bbox, 'area': area,
                   'iscrowd': 0, 'category_id': int(obj.category_id)}
            ann['segmentation'] = contours
            output['annotations'].append(ann)

        for page in tqdm(self.pages):
            img = cv2.imread(page.img_path)
            if img is None:
                print(f"Cannot load {page.img_path}")
                continue

            prefix = page.source_xml + "_" if add_prefix else ""
            file_name = prefix + str(Path(page.img_path).name)
            if Path(file_name).suffix != suffix:
                file_name = str(Path(file_name).with_suffix('.jpg'))
            image = {'file_name': file_name,
                     'width': int(fx * img.shape[1]), 'height': int(fy * img.shape[0]), "id": image_id}
            output['images'].append(image)
            for obj in page.objects:
                if add_block:
                    if isinstance(obj, NDLLine):
                        add_line_annotation(obj)
                    else:
                        add_annotation(obj)
                    instance_num[int(obj.category_id)] += 1
                    annotation_id += 1

            image_id += 1

        print(instance_num)

        output['categories'] = categories
        output['info'] = {
            "description": "NDL",
            "url": "",
            "version": "0.1a",
            "year": 2021,
            "contributor": "morpho",
            "date_created": "2021/09/01"
        }
        output['licenses'] = []
        return output

    def train_test_split(self, ratio: float = 0.9):
        import random
        from copy import deepcopy
        print("start train_test_split")
        pages = deepcopy(self.pages)
        random.shuffle(pages)
        split = int(ratio * len(pages))
        return NDLDataset(pages[:split]), NDLDataset(pages[split:])


def json_to_file(data, output_path: str):
    import json
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def main(xml_paths: List[str] = None, xml_list_path: str = None,
         img_dirs: List[str] = None,  img_list_path: str = None,
         show_summary: bool = False, fx: float = 1.0, fy: float = 1.0,
         train_json_path: str = "generated/train.json", test_json_path: str = "generated/test.json",
         add_prefix: bool = False):
    if xml_list_path is not None:
        xml_paths = list([s.strip() for s in open(xml_list_path).readlines()])
    if xml_paths is None:
        print('Please specify --xml_paths or --xml_list_path')
        return -1

    if img_list_path is not None:
        img_dirs = list([s.strip() for s in open(img_list_path).readlines()])
    if img_dirs is None:
        print('Please specify --img_dirs or --img_list_path')
        return -1

    dataset = NDLDataset()
    for xml_path, img_dir in zip(xml_paths, img_dirs):
        dataset.parse(xml_path, img_dir)
    if show_summary:
        dataset.summary()

    train_dataset, test_dataset = dataset.train_test_split()
    train_json = train_dataset.to_coco_fmt(fx=fx, fy=fy, add_prefix=add_prefix)
    json_to_file(train_json, train_json_path)
    test_json = test_dataset.to_coco_fmt(fx=fx, fy=fy, add_prefix=add_prefix)
    json_to_file(test_json, test_json_path)

    # whole data annotation
    import os
    data_json_path = os.path.join(
        os.path.dirname(train_json_path), 'data.json')
    data_json = dataset.to_coco_fmt(fx=fx, fy=fy, add_prefix=add_prefix)
    json_to_file(data_json, data_json_path)


if __name__ == '__main__':
    auto_run(main)
