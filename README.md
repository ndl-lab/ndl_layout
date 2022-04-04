# NDLレイアウト認識用リポジトリ

レイアウト要素を抽出するためのモジュールのリポジトリです。
本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。


# 環境構築
* 動作環境
  * Python3.6 で動作を確認しています。
  * 必要ライブラリは以下の通りです。
    * カスタマイズした [mmdetection](https://github.com/ndl-lab/mmdetection)を利用しています。


# 使い方
※スクリプトファイルはndl_layoutディレクトリ直下で実行すること

## tools/process.py : 推論用モジュール + CLI

学習結果を使って推論を実行する。学習済みのモデルを `ndl_layout/models` 下に配置する必要がある。

画像リストを引数で指定するには img_paths オプションを、画像リストをファイルから読み込む場合には list_path オプションを指定する。

output_path で出力 XML ファイルの格納先を変更することができる。（デフォルトは layout_prediction.xml）

use_show オプションを追加すると処理結果をGUI上で確認することができる。

img_pathsオプションで画像リストを指定する例
```bash
python -m tools.process --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml
```

list_path オプションで画像リストを指定する例
```bash
python -m tools.process --list_path image_list_file.list --use_show --output_path layout_prediction.xml
```

## tools/preprocess.py : 学習画像の追加＆変換

画像のファイル名の変換、縮小を行い、MS COCO 形式に整形。

```bash
python -m tools.preprocess images_data_dir output_dir --use_link
```

出力解像度を下げる必要がない場合には、`--use_link`オプションを指定する。

高解像の場合など、解像度を下げたい場合には `--use_shrink` を使うと画像サイズとアノテーションを半分のサイズに縮小して出力する。

本リポジトリの追加学習に使用可能なファイル(アノテーション情報の含まれるjson及び、前処理後の画像)は `output_dir` で指定したディレクトリに出力される。 


## 学習時の手順
1) ndl_layout/tools/preprocess.pyを使用し、NDL形式の画像とアノテーションファイル(xml)をCOCO形式に変換し保存する。
```
python -m tools.preprocess images_data_dir output_dir --use_link
```
output_dir内に画像のシンボリックリンク（またはコピー）とCOCO形式のアノテーションファイル(.json)を保存する。

アノテーションファイルは、data.json(全データのアノテーション)、train.json(ランダムに全体の9割)、test.json(train以外の残る1割)を生成する。

2) mmdetection/tools/train_ndl.py を使用し、モデルを学習する。
最終提出モデルのconfigは`configs/ndl/cascade_rcnn_r50_fpn_1x_ndl_1024_eql.py`
```
python tools/train_ndl.py configs/ndl/cascade_rcnn_r50_fpn_1x_ndl_1024_eql.py
```
学習データ、work directory、初期値、学習回数等はconfigファイル内で指定するか、train_ndl.pyのオプションを使用する。オプションで指定されたものが優先される。

work directoryに、学習したモデル(epoch_XX.pth または latest.pth)とconfigファイル(train_ndl.pyのオプションを使用した場合その内容も反映)、学習時のログファイル(.logと.log.json)が保存される。

なお、このリポジトリで公開しているモデルの学習時の初期重みには
https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth
を使用した。
