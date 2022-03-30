#!/usr/bin/env python

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from .utils import auto_run
from pathlib import Path


def iterfiles(root_dir: Path):
    """
    Iterate all subfiles under root_dir.
    This is not a pathlib implementation because pathlib does not support globs across symbols..
    """

    from glob import glob
    return [Path(p) for p in glob(f'{root_dir}/**', recursive=True)]


def main(root_dir: str, output_dir: str, use_link: bool = False, use_shrink: bool = False):
    import os
    import subprocess
    from pathlib import Path

    def call(*args):
        print(*args)
        subprocess.call(args, stderr=None)

    scale = 1.0
    if use_shrink:
        scale = 0.5

    os.makedirs(output_dir, exist_ok=True)

    if use_link:
        for path in iterfiles(root_dir):
            if path.is_dir() and (path / "img").exists() and (path / "xml").exists():
                for img_path in (path / "img").iterdir():
                    if img_path.is_file() and not str(img_path.name).startswith('._'):
                        if img_path.suffix == '.jpg':
                            call('ln', '-s', str(img_path.resolve()),
                                 str(Path(output_dir) / (path.name + "_" + img_path.name)))
                        else:
                            import cv2
                            img = cv2.imread(str(img_path.resolve()))
                            dst_path = (
                                Path(output_dir) / (path.name + "_" + img_path.name)).with_suffix('.jpg')
                            cv2.imwrite(str(dst_path), img)

    if use_shrink:
        call('python', './tools/shrink2.py', output_dir, output_dir + "_half")

    xml_paths = []
    img_dirs = []

    for path in iterfiles(root_dir):
        if path.is_dir() and (path / "img").exists() and (path / "xml").exists():
            xml_path = [p for p in (
                path / "xml").iterdir() if p.suffix == ".xml" and not p.name.startswith('._')]
            assert len(xml_path) == 1
            xml_paths.append(str(xml_path[0]))
            img_dirs.append(str(path / "img"))

    # xml_paths = xml_paths[0:1]
    # img_dirs = img_dirs[0:1]

    dataset_name = Path(root_dir).name
    train_json_path = f'generated/{dataset_name}_train.json'
    test_json_path = f'generated/{dataset_name}_test.json'
    if use_link:
        if use_shrink:
            train_json_path = str(Path(output_dir + "_half") / "train.json")
            test_json_path = str(Path(output_dir + "_half") / "test.json")
        else:
            train_json_path = str(Path(output_dir) / "train.json")
            test_json_path = str(Path(output_dir) / "test.json")
    print("train_json_path :", train_json_path)
    print("test_json_path :", test_json_path)

    cmd = ['python', '-m', 'tools.ndl_parser', '--xml_paths', *xml_paths, '--img_dirs', *img_dirs, '--add_prefix', 'True',
           '--fx', str(scale), '--fy', str(scale), '--train_json_path', train_json_path, '--test_json_path', test_json_path]
    cmd_len = sum(len(line) for line in cmd)
    if cmd_len > 2_000_000:
        xml_list_filename = 'tmp_xml_paths.list'
        img_list_filename = 'tmp_img_dirs.list'
        print('xml_paths or img_dirs is long.')
        print('export xml_paths and img_dirs to {} and {}'.format(
            xml_list_filename, img_list_filename))
        with open(xml_list_filename, 'w') as f:
            for path in xml_paths:
                f.write(path+'\n')
        with open(img_list_filename, 'w') as f:
            for path in img_dirs:
                f.write(path+'\n')
        call('python', './tools/ndl_parser.py', '--xml_list_path', xml_list_filename, '--img_list_path', img_list_filename, '--add_prefix',
             'True', '--fx', str(scale), '--fy', str(scale), '--train_json_path', train_json_path, '--test_json_path', test_json_path)
    else:
        call('python', './tools/ndl_parser.py', '--xml_paths', *xml_paths, '--img_dirs', *img_dirs, '--add_prefix', 'True',
             '--fx', str(scale), '--fy', str(scale), '--train_json_path', train_json_path, '--test_json_path', test_json_path)


auto_run(main)
