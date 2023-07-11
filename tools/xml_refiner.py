#!/usr/bin/env python

# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import argparse
import xml.etree.ElementTree as ET

super_types = ['TEXTBLOCK']


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('xml', help='input xml file path')
    parser.add_argument('-o', '--out', default=None,
                        help='output xml file path')
    parser.add_argument('-ov', '--vh_overlap_th', type=int, default=2,
                        help='How many intersecting vertical and horizontal boxes should be removed')
    parser.add_argument('-im', '--inclusion_margin', default=0.05,
                        help='inclusion margin ratio. default 0.05')
    parser.add_argument('-co', '--category_option', default='SAME',
                        help='SAME(default) : investigate whether inclusion is only for the same category\n'
                             'SIM  : investigate inclusion for similar categories(line/block).\n'
                             'ALL  : investigate category-independent inclusions.\n')
    parser.add_argument('--rm_vh_confusion_only', action='store_true')
    parser.add_argument('--rm_inclusion_only', action='store_true')
    return parser.parse_args()


def get_points(elm):
    x1 = int(elm.attrib['X'])
    y1 = int(elm.attrib['Y'])
    x2 = x1 + int(elm.attrib['WIDTH'])
    y2 = y1 + int(elm.attrib['HEIGHT'])

    return x1, y1, x2, y2


def vh_comp(elm_a, elm_b):
    v1 = int(elm_a.attrib['WIDTH'])-int(elm_a.attrib['HEIGHT'])
    v2 = int(elm_b.attrib['WIDTH'])-int(elm_b.attrib['HEIGHT'])
    return v1*v2 > 0


def vh_overlapping(elm_a, elm_b):
    if vh_comp(elm_a, elm_b):
        # vert vert or hori hori
        return False
    else:
        a_x1, a_y1, a_x2, a_y2 = get_points(elm_a)
        b_x1, b_y1, b_x2, b_y2 = get_points(elm_b)
        # c ... intersection
        c_x1 = max(a_x1, b_x1)
        c_y1 = max(a_y1, b_y1)
        c_x2 = min(a_x2, b_x2)
        c_y2 = min(a_y2, b_y2)
        if (c_x1 > c_x2) or (c_y1 > c_y2):
            return False  # No intersection
        else:
            return True

def is_overlapped(elm, page, overlap_th):
    if elm.tag != 'LINE':
        return False
    vh_overlap_count = 0
    overlapped = False
    for elm_ref in page:
        if elm_ref.tag in super_types: # elm_ref is TEXTBLOCK
            for sub_elm_ref in elm_ref:
                if sub_elm_ref.tag == 'SHAPE':
                    continue
                if sub_elm_ref.tag == 'LINE' and elm.attrib['TYPE'] == sub_elm_ref.attrib['TYPE']:
                    if vh_overlapping(elm, sub_elm_ref):
                        vh_overlap_count += 1
                        if vh_overlap_count >= overlap_th:
                            return True
        elif elm_ref.tag == 'BLOCK' and elm_ref.attrib['TYPE'] == '広告':
            for sub_elm_ref in elm_ref:
                if sub_elm_ref.tag == 'TEXTBLOCK':
                    for ssub_elm_ref in reversed(sub_elm_ref):
                        if ssub_elm_ref.tag == "SHAPE":
                            continue
                        if ssub_elm_ref.tag == 'LINE' and elm.attrib['TYPE'] == ssub_elm_ref.attrib['TYPE']:
                            if vh_overlapping(elm, ssub_elm_ref):
                                vh_overlap_count += 1
                                if vh_overlap_count >= overlap_th:
                                    return True
                 # LINE in block ad
                elif sub_elm_ref.tag == 'LINE' and elm.attrib['TYPE'] == sub_elm_ref.attrib['TYEP']:
                    if vh_overlapping(elm, sub_elm_ref):
                        vh_overlap_count += 1
                        if vh_overlap_count >= overlap_th:
                            return True
        # elm_ref is LINE
        elif elm.tag == elm_ref.tag and elm.attrib['TYPE'] == elm_ref.attrib['TYPE']:
            if vh_overlapping(elm, elm_ref):
                vh_overlap_count += 1
                if vh_overlap_count >= overlap_th:
                    return True

    return overlapped

def refine_vh_confusion(root, overlap_th):
    print('Refine VH Confusion')
    for page in root:
        print(page.attrib['IMAGENAME'])

        for elm in reversed(page):
            # vh overlap count
            if elm.tag in super_types: # tag is TEXTBLOCK
                for sub_elm in reversed(elm):
                    if sub_elm.tag == 'SHAPE':
                        continue
                    if is_overlapped(sub_elm, page, overlap_th):
                        elm.remove(sub_elm)
            elif elm.tag == 'BLOCK' and elm.attrib['TYPE'] == '広告':
                for sub_elm in reversed(elm):
                    if sub_elm.tag in super_types: # if TEXTBLOCK
                        for ssub_elm in reversed(sub_elm):
                            if sub_elm.tag == 'SHAPE':
                                continue
                            # line in textblock in block_ad
                            if is_overlapped(ssub_elm, page, overlap_th):
                                sub_elm.remove(ssub_elm)
                                break
                    else: # Lines in block_ad and outside textblock
                        if is_overlapped(sub_elm, page, overlap_th):
                            elm.remove(sub_elm)
                            break
            else :
                if is_overlapped(elm, page, overlap_th):
                    page.remove(elm)
    return root


def include(child, parent, margin=0.05):
    p_x1, p_y1, p_x2, p_y2 = get_points(parent)
    c_x1, c_y1, c_x2, c_y2 = get_points(child)
    if p_x1 == c_x1 and p_y1 == c_y1 and p_x2 == c_x2 and p_y2 == c_y2: # perfect same
        return False

    w_m = int(child.attrib['WIDTH']) * margin
    h_m = int(child.attrib['HEIGHT']) * margin

    if (p_x1-w_m <= c_x1) and (p_y1-h_m <= c_y1) and (p_x2+w_m >= c_x2) and (p_y2+h_m > c_y2):
        return True
    else:
        return False

def is_included(elm, page, margin=0.05, category_option='SAME'):
    include_flag = False
    for elm_ref in page:  # parent
        if elm_ref.tag in super_types: # TEXTBLOCK
            for sub_elm_ref in reversed(elm_ref): # elm in TEXTBLOCK
                if sub_elm_ref.tag == 'SHAPE':
                    continue
                if category_option == 'SAME':
                    if elm.attrib['TYPE'] != sub_elm_ref.attrib['TYPE']:
                        continue
                elif category_option == 'SIM':
                    if elm.tag != sub_elm_ref.tag:
                        continue
                if include(child=elm, parent=sub_elm_ref, margin=margin):
                    return True
        elif elm_ref.tag == "BLOCK" and elm_ref.attrib['TYPE']=='広告': # block_ad
            for sub_elm_ref in reversed(elm_ref):
                if sub_elm_ref.tag in super_types: # textblock in block_ad
                    for ssub_elm_ref in reversed(sub_elm_ref): # Lines in textblock in block_ad
                        if ssub_elm_ref.tag == "SHPAE":
                            continue
                        if category_option == 'SAME':
                            if elm.attrib['TYPE'] != ssub_elm_ref.attrib['TYPE']:
                                continue
                        elif category_option == 'SIM':
                            if elm.tag != ssub_elm_ref.tag:
                                continue
                        if inlude(child=elm, parent=ssub_elm_ref, margin=margin):
                            return True
        else: # not TEXTBLOCK nor block_ad
            if category_option == 'SAME':
                if elm.attrib['TYPE'] != elm_ref.attrib['TYPE']:
                    continue
            elif category_option == 'SIM':
                if elm.tag != elm_ref.tag:
                    continue
            if include(child=elm, parent=elm_ref, margin=margin):
                return True

    return include_flag

def refine_inclusion(root, margin=0.05, category_option='SAME'):
    print('Refine inclusion')
    for page in root:
        print(page.attrib['IMAGENAME'])
        for elm in reversed(page):  # child
            if elm.tag in super_types: # if TEXTBLOCK
                for sub_elm in reversed(elm):
                    if sub_elm.tag == 'SHAPE': # skip shape info
                        continue
                    # include_flag = False
                    if is_included(sub_elm, page, margin=margin, category_option=category_option):
                        elm.remove(sub_elm)
                        break
            elif elm.tag == "BLOCK" and elm.attrib['TYPE']=='広告':
                for sub_elm in reversed(elm):
                    if sub_elm.tag in super_types: # if TEXTBLOCK
                        for ssub_elm in reversed(sub_elm):
                            if sub_elm.tag == 'SHAPE':
                                continue
                            # line in textblock in block_ad
                            if is_included(ssub_elm, page, margin=margin, category_option=category_option):
                                sub_elm.remove(ssub_elm)
                                break
                    else: # Lines in block_ad and outside textblock
                        if is_included(sub_elm, page, margin=margin, category_option=category_option):
                            elm.remove(sub_elm)
                            break
            else:
                # include_flag = False
                if is_included(elm, page, margin=margin, category_option=category_option):
                    page.remove(elm)
                    break
    return root


def refine(xml, out_xml, vh_overlap_th=2, margin=0.05, category_option='SAME', vh=True, inc=True):
    tree = ET.parse(xml)
    root = tree.getroot()
    if vh:
        root = refine_vh_confusion(root, vh_overlap_th)
    if inc:
        root = refine_inclusion(root, margin, category_option)

    tree.write(out_xml, encoding='UTF-8')
    return


def main():
    args = parse_args()
    out_xml_path = 'out.xml'
    if args.out is not None:
        out_xml_path = args.out
    refine(xml=args.xml,
           out_xml=out_xml_path,
           vh_overlap_th=args.vh_overlap_th,
           margin=args.inclusion_margin,
           category_option=args.category_option,
           vh=not args.rm_inclusion_only,
           inc=not args.rm_vh_confusion_only)

    print('Export: {}'.format(out_xml_path))


if __name__ == '__main__':
    main()
