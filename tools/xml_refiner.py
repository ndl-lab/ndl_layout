#!/usr/bin/env python

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import argparse
import xml.etree.ElementTree as ET


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


def refine_vh_confusion(root, overlap_th):
    print('Refine VH Confusion')
    for page in root:
        print(page.attrib['IMAGENAME'])

        for elm in reversed(page):
            # vh overlap count
            vh_overlap_count = 0
            for elm_ref in page:
                if elm.tag == 'LINE' and elm.tag == elm_ref.tag and elm.attrib['TYPE'] == elm_ref.attrib['TYPE']:
                    if vh_overlapping(elm, elm_ref):
                        vh_overlap_count += 1
                if vh_overlap_count >= overlap_th:
                    page.remove(elm)
                    break
    return root


def include(parent, child, margin=0.05):
    p_x1, p_y1, p_x2, p_y2 = get_points(parent)
    c_x1, c_y1, c_x2, c_y2 = get_points(child)
    if p_x1 == c_x1 and p_y1 == c_y1 and p_x2 == c_x2 and p_y2 == c_y2:
        return False

    w_m = int(child.attrib['WIDTH']) * margin
    h_m = int(child.attrib['HEIGHT']) * margin

    if (p_x1-w_m <= c_x1) and (p_y1-h_m <= c_y1) and (p_x2+w_m >= c_x2) and (p_y2+h_m > c_y2):
        return True
    else:
        return False


def refine_inclusion(root, margin=0.05, category_option='SAME'):
    print('Refine inclusion')
    for page in root:
        print(page.attrib['IMAGENAME'])
        for elm in reversed(page):  # child
            include_flag = False
            for elm_ref in page:  # parent
                if category_option == 'SAME':
                    if elm.attrib['TYPE'] != elm_ref.attrib['TYPE']:
                        continue
                elif category_option == 'SIM':
                    if elm.tag != elm_ref.tag:
                        continue

                include_flag = include(parent=elm_ref, child=elm, margin=margin)
                if include_flag:
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
