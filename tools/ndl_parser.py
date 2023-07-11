#!/usr/bin/env python

# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import List
from enum import IntEnum, auto


class Category(IntEnum):
    LINE_MAIN = 0
    LINE_INOTE = auto()
    LINE_HNOTE = auto()
    LINE_CAPTION = auto()
    LINE_AD = auto()
    BLOCK_FIG = auto()
    BLOCK_TABLE = auto()
    BLOCK_PILLAR = auto()
    BLOCK_FOLIO = auto()
    BLOCK_RUBI = auto()
    BLOCK_CHART = auto()
    BLOCK_EQN = auto()
    BLOCK_CFM = auto()
    BLOCK_ENG = auto()
    BLOCK_AD = auto()
    TEXT_BLOCK = auto()
    TEXT_BLOCK_AD = auto()
    CHAR = auto()
    NUM = auto()


class InlineCategory(IntEnum):
    INLINE_ENG = auto()
    INLINE_RENG = auto()
    INLINE_COLOR = auto()
    INLINE_EQN = auto()
    INLINE_CFM = auto()
    INLINE_HINV = auto()
    INLINE_HAND = auto()


categories = [
    {'id': int(Category.LINE_MAIN),     'name': 'line_main',     'org_name': '本文'},
    {'id': int(Category.LINE_INOTE),    'name': 'line_inote',    'org_name': '割注'},
    {'id': int(Category.LINE_HNOTE),    'name': 'line_hnote',    'org_name': '頭注'},
    {'id': int(Category.LINE_CAPTION),  'name': 'line_caption',  'org_name': 'キャプション'},
    {'id': int(Category.LINE_AD),       'name': 'line_ad',       'org_name': '広告文字'},
    {'id': int(Category.BLOCK_FIG),     'name': 'block_fig',     'org_name': '図版'},
    {'id': int(Category.BLOCK_TABLE),   'name': 'block_table',   'org_name': '表組'},
    {'id': int(Category.BLOCK_PILLAR),  'name': 'block_pillar',  'org_name': '柱'},
    {'id': int(Category.BLOCK_FOLIO),   'name': 'block_folio',   'org_name': 'ノンブル'},
    {'id': int(Category.BLOCK_RUBI),    'name': 'block_rubi',    'org_name': 'ルビ'},
    {'id': int(Category.BLOCK_CHART),   'name': 'block_chart',   'org_name': '組織図'},
    {'id': int(Category.BLOCK_EQN),     'name': 'block_eqn',     'org_name': '数式'},
    {'id': int(Category.BLOCK_CFM),     'name': 'block_cfm',     'org_name': '化学式'},
    {'id': int(Category.BLOCK_ENG),     'name': 'block_eng',     'org_name': '欧文'},
    {'id': int(Category.BLOCK_AD),      'name': 'block_ad',      'org_name': '広告'},
    {'id': int(Category.TEXT_BLOCK),    'name': 'text_block',    'org_name': '本文ブロック'},
    {'id': int(Category.TEXT_BLOCK_AD), 'name': 'text_block_ad', 'org_name': '広告本文ブロック'},
    {'id': int(Category.CHAR),          'name': 'char',          'org_name': 'char'},
    {'id': int(Category.NUM),           'name': 'void',          'org_name': 'void'}]

categories_org_name_index = {elem['org_name']: elem for elem in categories}
categories_name_index = {elem['name']: elem for elem in categories}


def org_name_to_id(s: str):
    return categories_org_name_index[s]['id']


def name_to_org_name(s: str):
    return categories_name_index[s]['org_name']


inline_categories = [
    {'id': int(InlineCategory.INLINE_ENG),   'name': 'inline_eng',   'org_name': '欧文'},
    {'id': int(InlineCategory.INLINE_RENG),  'name': 'inline_reng',  'org_name': '回転欧文'},
    {'id': int(InlineCategory.INLINE_COLOR), 'name': 'inline_color', 'org_name': '色付文字'},
    {'id': int(InlineCategory.INLINE_EQN),   'name': 'inline_eqn',   'org_name': '数式'},
    {'id': int(InlineCategory.INLINE_CFM),   'name': 'inline_cfn',   'org_name': '化学式'},
    {'id': int(InlineCategory.INLINE_HINV),  'name': 'inline_hinv',  'org_name': '縦中横'},
    {'id': int(InlineCategory.INLINE_HAND),  'name': 'inline_hand',  'org_name': '手書き'}]

inline_categories_org_name_index = {elem['org_name']: elem for elem in inline_categories}
inline_categories_name_index = {elem['name']: elem for elem in inline_categories}


def inline_org_name_to_id(s: str):
    return inline_categories_org_name_index[s]['id']


def inline_name_to_org_name(s: str):
    return inline_categories_name_index[s]['org_name']


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


class NDLInline(NDLObject):
    def __init__(self, type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_id = inline_org_name_to_id(type)
        self.type = type

    def __repr(self):
        return f'NDLInline({self.type}, {self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


class NDLLine(NDLObject):
    def __init__(self, chars: List[NDLChar], opt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chars = chars
        self.category_id = org_name_to_id(opt)
        self.opt = opt

    def __repr__(self):
        return f'NDLLine({self.chars}, {self.opt}, {self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


class NDLTextblock(NDLObject):
    def __init__(self, points, opt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_id = org_name_to_id(opt)
        self.type = opt
        self.points = points

    def __repr__(self):
        return f'NDLTextblock({self.type}, {self.x}, {self.y}, {self.width}, {self.height}, category_id={self.category_id})'


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

        def get_tag(elem):
            prefix, has_namespace, postfix = elem.tag.partition('}')
            if has_namespace:
                tag = postfix
            else:
                tag = child_elem.tag
            return tag

        def parse_points(elem):
            points_str = elem[0].attrib['POINTS']
            points_list = [float(i) for i in points_str.split(',')]
            return points_list

        def points_to_bbox(points_list):
            if len(points_list) % 2 == 1:
                print("ERROR: Invalid polygon points")
                return 0, 0, 0, 0
            it = iter(points_list)
            min_x = points_list[0]
            min_y = points_list[1]
            max_x = 0
            max_y = 0
            for i in range(2, len(it), 2):
                tx = it[i]
                ty = it[i+1]
                if min_x > tx:
                    min_x = tx
                if max_x < tx:
                    max_x = tx
                if min_y > ty:
                    min_y = ty
                if max_y < ty:
                    max_y = ty
            return min_x, min_y, (max_x-min_x), (max_y-min_y)

        def parse_textblock(elem, objects, ad=False):
            for child_elem in elem:
                prefix, has_namespace, postfix = child_elem.tag.partition('}')
                if has_namespace:
                    tag = postfix
                else:
                    tag = child_elem.tag

                if tag == 'SHAPE':  # SHAPE info in TEXT_BLOCK
                    points = parse_points(child_elem)
                    bbox = points_to_bbox(points)
                    opt = '本文ブロック'
                    if ad:
                        opt = '広告本文ブロック'
                    objects.append(
                        NDLTextblock(points, opt, *bbox))
                elif tag == 'LINE':  # LINE in TEXT_BLOCK
                    chars = []
                    bbox = parse_bbox(child_elem)
                    for char in child_elem:
                        bbox_char = parse_bbox(char)
                        if get_tag(char) != 'CHAR':  # INLINE
                            print(char.attrib['TYPE'])
                            chars.append(NDLInline(char.attrib['TYPE'], *bbox_char))
                        else:  # CHAR
                            chars.append(NDLChar(char.attrib['MOJI'], *bbox_char))
                    objects.append(
                        NDLLine(chars, child_elem.attrib.get('TYPE', ''), *bbox))
                else:
                    continue

        for page in root:
            img_path = str(Path(img_dir) / page.attrib['IMAGENAME'])
            objects = []
            # print('\n**len {}'.format(len(page)))
            for elem in page:
                prefix, has_namespace, postfix = elem.tag.partition('}')
                if has_namespace:
                    tag = postfix
                else:
                    tag = elem.tag

                if elem.get('ERROR') is not None:
                    print('ERROR attrib is found!!!!')
                    print('skip the element')
                    continue

                if tag == 'BLOCK':
                    bbox = parse_bbox(elem)
                    objects.append(NDLBlock(elem.attrib['TYPE'], *bbox))
                    if elem.attrib['TYPE'] == '広告':
                        print("parse_ad")
                        for child_elem in elem:
                            if get_tag(child_elem) == 'TEXTBLOCK':
                                parse_textblock(child_elem, objects, ad=True)

                elif tag == 'LINE':
                    chars = []
                    bbox = parse_bbox(elem)
                    for char in elem:
                        bbox_char = parse_bbox(char)
                        if get_tag(char) != 'CHAR':  # INLINE
                            print(char.attrib['TYPE'])
                            chars.append(NDLInline(char.attrib['TYPE'], *bbox_char))
                        else:  # CHAR
                            chars.append(NDLChar(char.attrib['MOJI'], *bbox_char))
                    objects.append(
                        NDLLine(chars, elem.attrib.get('TYPE', ''), *bbox))
                elif tag == 'TEXTBLOCK':
                    parse_textblock(elem, objects)
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
        import numpy as np
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

        def make_contours(obj):
            x1, y1 = fx * obj.x, fy * obj.y
            width, height = fx * obj.width, fy * obj.height
            bbox = [x1, y1, width, height]
            it = iter(obj.points)
            cv_contours = []
            for tx, ty in zip(*[it]*2):
                tmp = np.array([tx, ty], dtype='float32')
                cv_contours.append(tmp)
            area = cv2.contourArea(np.array(cv_contours))
            contour = obj.points
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

            # chars as line
            _, contour, area = make_bbox(obj)
            contours.append(contour)
            if area == 0:
                area = area_sum

            ann = {'image_id': image_id, 'id': annotation_id, 'bbox': bbox, 'area': area,
                   'iscrowd': 0, 'category_id': int(obj.category_id)}
            ann['segmentation'] = contours
            output['annotations'].append(ann)

        def add_textblock_annotation(obj):
            bbox, contour, area = make_contours(obj)
            ann = {'image_id': image_id, 'id': annotation_id, 'bbox': bbox, 'area': area,
                   'iscrowd': 0, 'category_id': int(obj.category_id)}
            ann['segmentation'] = [contour]
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
                    elif isinstance(obj, NDLTextblock):
                        add_textblock_annotation(obj)
                    else:  # BLOCK
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
    from utils import auto_run
    auto_run(main)
