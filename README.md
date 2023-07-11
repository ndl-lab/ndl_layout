# NDLレイアウト認識用リポジトリ

レイアウト要素を抽出するモジュールのリポジトリです。
本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。
本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については LICENSEをご覧ください。

※スクリプトファイルはトップディレクトリで実行する必要があることに注意。

* 動作環境
  * Python3.8 で動作を確認しています。
  * `tools/process_textblock.py` は実行する計算機のスペック、入力画像によってはメモリ不足になる場合があります。メモリ不足になる場合、 
  `mmdetection` の [`mmdetection/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py`](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L19) の以下の部分の `GPU_MEM_LIMIT` の値を小さくすることで回避できます。
  ```
  # TODO: This memory limit may be too much or too little. It would be better to
  # determine it based on available resources.
  GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit
  ```


* `tools/ndl_parser.py`

NDL の xml をパースして読み込む用のモジュール。

* `tools/preprocess.py` : 学習画像の追加＆変換

画像のファイル名の変換、縮小を行い、MS COCO 形式に整形。

```
python -m tools.preprocess images_data_dir output_dir --use_link
```

基本的には`--use_link`オプションを指定、高解像の場合などは `--use_shrink` を使うと画像サイズとアノテーションを半分のサイズに縮小して出力
学習に使用するファイルは `output_dir` に出力。 (json と 前処理された image)


* `tools/process.py` : 推論用モジュール

学習済みモデルを使って推論を実行。
学習済みのモデルは `ndl_layout/models` 下に下記のコマンドで配置する（または推論時に--checkpointオプションで指定可能）。
`
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/ndl_layout/epoch_375.pth -P ./ndl_layout/models
`
この推論スクリプトは、Detection（矩形の検出）のみ行うモデル（Cascade RCNN等）専用。
DetectionとInstance Segmentataionを同時に行うモデル（Cascade Mask RCNN等）を使用する場合は後述の `tools/process_textblock.py` を使用する。

画像リストを引数で指定するには `img_paths` オプションを、画像リストをファイルから読み込む場合には `list_path` オプションを指定する。

`output_path` に出力 XML ファイルの格納先を指定可能。（デフォルトは `layout_prediction.xml`）

`use_show` オプションを追加すると処理結果のGUIでの確認も可能。

`img_paths` 使用の場合は
```
python -m tools.process --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml
```

一方で `list_path` 使用の場合は
```
python -m tools.process --list_path image_list_file.list --use_show --output_path layout_prediction.xml
```



* `tools/process_textblock.py` : 推論用モジュール

使用方法、オプションは tools/process.py と同様。
この推論スクリプトは、 Detection と Instance Segmentataion を同時に行うモデル（Cascade Mask RCNN等）用。本スクリプトでは、多数のDetection と Instance Segmentataionを行うため、実行する計算機のスペック、入力画像によってはメモリ不足になる場合があります。メモリ不足になる場合、 
`mmdetection` の [`mmdetection/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py`](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L19) の `GPU_MEM_LIMIT` の値を小さくすることで回避できます。


img_paths 使用の場合は
```
python -m tools.process_textblock --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml
```

一方で list_path 使用の場合は
```
python -m tools.process_textblock --list_path image_list_file.list --use_show --output_path layout_prediction.xml
```
