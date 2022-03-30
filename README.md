# NDLレイアウト認識用リポジトリ

レイアウト要素を抽出するプロジェクトのリポジトリです。
本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。
※スクリプトファイルはトップディレクトリで実行する必要があることに注意してください。

* 動作環境
  * Python3.6 で動作を確認しています。
  * 必要ライブラリは以下の通りです。
    * カスタマイズした [mmdetection](https://github.com/ndl-lab/mmdetection)

* tools/ndl_parser.py

NDL の xml をパースして読み込む用のモジュール。

* tools/preprocess.py : 学習画像の追加＆変換

画像のファイル名の変換、縮小を行い、MS COCO 形式に整形。

```
python -m tools.preprocess images_data_dir output_dir --use_link
```

基本的には`--use_link`オプションを指定、高解像の場合などは `--use_shrink` を使うと画像サイズとアノテーションを半分のサイズに縮小して出力
学習に使用するファイルは `output_dir` に出力。 (json と 前処理された image)

* tools/process.py : 推論用モジュール + CLI

学習結果を使って推論を実行。学習済みのモデルを `ndl_layout/models` 下に配置する必要あり。

画像リストを引数で指定するには img_paths オプションを、画像リストをファイルから読み込む場合には list_path オプションを指定する。

output_path に出力 XML ファイルの格納先を指定可能。（デフォルトは layout_prediction.xml）

use_show オプションを追加すると処理結果のGUIでの確認も可能。

img_paths 使用の場合は
```
python -m tools.process --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml
```

一方で list_path 使用の場合は
```
python -m tools.process --list_path image_list_file.list --use_show --output_path layout_prediction.xml
```

* 学習方法
1) ndl_layout/tools/preprocess.pyを使用し、NDL形式の画像とアノテーションファイル(xml)をCOCO形式に変換し保存する。
```
python -m tools.preprocess images_data_dir output_dir --use_link
```
output_dir内に画像のシンボリックリンク（またはコピー）とCOCO形式のアノテーションファイル(.json)を保存する。アノテーションファイルは、data.json(全データのアノテーション)、train.json(ランダムに全体の9割)、test.json(train以外の残る1割)を生成する。

2) mmdetection/tools/train_ndl.py を使用し、モデルを学習する。
最終提出モデルのconfigは`configs/ndl/cascade_rcnn_r50_fpn_1x_ndl_1024_eql.py`
```
python tools/train_ndl.py configs/ndl/cascade_rcnn_r50_fpn_1x_ndl_1024_eql.py
```
学習データ、work directory、初期値、学習回数等はconfigファイル内で指定するか、train_ndl.pyのオプションを使用する。オプションで指定されたものが優先される。

work directoryに、学習したモデル(epoch_XX.pth または latest.pth)とconfigファイル(train_ndl.pyのオプションを使用した場合その内容も反映)、学習時のログファイル(.logと.log.json)が保存される。

初期値には
https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth
を使用した。

(ndl_layout/tools/process.pyによる推論時は、2)でwork directoryに生成されたモデル(.pth)とconfigを使用する。）
