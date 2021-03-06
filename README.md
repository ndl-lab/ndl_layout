# NDLOCR用レイアウト認識モジュール

レイアウト要素を抽出するためのモジュールのリポジトリです。

本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については
[LICENSE](./LICENSE
)をご覧ください。

# 環境構築

python3.7かつ、cuda 11.1をインストール済みの環境の場合
ndl_layoutディレクトリ直下で以下のコマンドを実行する。
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
wget https://lab.ndl.go.jp/dataset/ndlocr/ndl_layout/ndl_layout_config.py -P ./models
wget https://lab.ndl.go.jp/dataset/ndlocr/ndl_layout/epoch_140_all_eql_bt.pth -P ./models
```

くわえて、元リポジトリ(https://github.com/open-mmlab/mmdetection)
をカスタマイズした[mmdetection](https://github.com/ndl-lab/mmdetection)
に依存しているため、下記のようにリポジトリの追加とインストールを行う。

```bash
git clone https://github.com/ndl-lab/mmdetection
cd mmdetection
python setup.py bdist_wheel
pip install dist/*.whl
```


# 使い方
※スクリプトファイルはndl_layoutディレクトリ直下で実行すること

## tools/process.py : 推論用モジュール + CLI

学習結果を使って推論を実行する。学習済みのモデルは`ndl_layout/models` 以下にあるものとする。

画像リストを引数で指定するには img_paths オプションを、画像リストをファイルから読み込む場合には list_path オプションを指定する。

output_path で出力 XML ファイルの格納先を変更することができる。（デフォルトは layout_prediction.xml）

use_show オプションを追加すると処理結果をGUI上で確認することができる。

img_pathsオプションで画像リストを指定する例
```bash
python -m tools.process --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml --config ./models/ndl_layout_config.py --checkpoint ./models/epoch_140_all_eql_bt.pth
```

list_path オプションで画像リストを指定する例
```bash
python -m tools.process --list_path image_list_file.list --use_show --output_path layout_prediction.xml --config ./models/ndl_layout_config.py --checkpoint ./models/epoch_140_all_eql_bt.pth
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
1) ndl_layout/tools/preprocess.pyを使用し、NDLOCRXMLDataset形式の画像とアノテーションファイル(xml)をCOCO形式に変換し保存する。
```
cd mmdetection
python -m tools.preprocess images_data_dir output_dir --use_link
```
output_dir内に画像のシンボリックリンク（またはコピー）とCOCO形式のアノテーションファイル(.json)を保存する。

アノテーションファイルは、data.json(全データのアノテーション)、train.json(ランダムに全体の9割)、test.json(train以外の残る1割)を生成する。

2) mmdetection/tools/train_ndl.py を使用し、モデルを学習する。
```
cd mmdetection
python tools/train_ndl.py configs/ndl/cascade_rcnn_r50_fpn_1x_ndl_1024_eql.py
```
学習データ、work directory、初期値、学習回数等はconfigファイル内で指定するか、train_ndl.pyのオプションを使用する。オプションで指定されたものが優先される。

work directoryに、学習したモデル(epoch_XX.pth または latest.pth)とconfigファイル(train_ndl.pyのオプションを使用した場合その内容も反映)、学習時のログファイル(.logと.log.json)が保存される。

なお、このリポジトリで公開しているモデル（設定ファイルは`configs/ndl/cascade_rcnn_r50_fpn_1x_ndl_1024_eql.py`を参照）の学習時の初期重みには
https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth
を使用した。
