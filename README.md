# ndl_layout(NDLOCR(ver.2.1)用レイアウト認識モジュール)

レイアウト要素を抽出するモジュールのリポジトリです。

本プログラムは、令和4年度NDLOCR追加開発事業の成果物である[ver.2.0](https://github.com/ndl-lab/ndlocr_cli/tree/ver.2.0)に対して、国立国会図書館が改善作業を行ったプログラムです。

事業の詳細については、[令和4年度NDLOCR追加開発事業及び同事業成果に対する改善作業](https://lab.ndl.go.jp/data_set/r4ocr/r4_software/)をご覧ください。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については LICENSEをご覧ください。

※スクリプトファイルはトップディレクトリで実行する必要があることに注意してください。

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

NDL の xml をパースして読み込む用のモジュールです。

* `tools/preprocess.py` : 学習画像の追加＆変換

画像のファイル名の変換、縮小を行い、MS COCO 形式に整形する処理は次のように実行可能です。

```
python -m tools.preprocess images_data_dir output_dir --use_link
```

基本的には`--use_link`オプションを指定します。高解像の場合などは `--use_shrink` を使うと画像サイズとアノテーションを半分のサイズに縮小して出力します。
学習に使用するファイルは `output_dir` に出力します。 (json と 前処理された image)


* `tools/process.py` : 推論用モジュール

学習済みモデルを使って推論処理を実行します。
学習済みのモデルは `ndl_layout/models` 下に下記のコマンドで配置する（または推論時に--checkpointオプションで指定）必要があります。

```
wget -nc https://lab.ndl.go.jp/dataset/ndlocr_v2/ndl_layout/ndl_retrainmodel.pth -P ./ndl_layout/models
```

この推論スクリプトは、Detection（矩形の検出）のみ行うモデル（Cascade RCNN等）専用です。
DetectionとInstance Segmentataionを同時に行うモデル（Cascade Mask RCNN等）を使用する場合は後述の `tools/process_textblock.py` を使用します。

画像リストを引数で指定するには `img_paths` オプションを、画像リストをファイルから読み込む場合には `list_path` オプションを指定します。

`output_path` に出力 XML ファイルの格納先を指定可能です。（デフォルトは `layout_prediction.xml`）

`use_show` オプションを追加すると処理結果のGUIでの確認も可能です。

`img_paths` 使用の場合は次のように実行します。
```
python -m tools.process --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml
```

`list_path` 使用の場合は次のように実行します。
```
python -m tools.process --list_path image_list_file.list --use_show --output_path layout_prediction.xml
```



* `tools/process_textblock.py` : 推論用モジュール

使用方法、オプションは tools/process.py と同様です。
この推論スクリプトは、 Detection と Instance Segmentataion を同時に行うモデル（Cascade Mask RCNN等）用。本スクリプトでは、多数のDetection と Instance Segmentataionを行うため、実行する計算機のスペック、入力画像によってはメモリ不足になる場合があります。メモリ不足になる場合、 
`mmdetection` の [`mmdetection/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py`](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L19) の `GPU_MEM_LIMIT` の値を小さくすることで回避できます。


処理対象をパスで指定する場合は次のように実行します。
```
python -m tools.process_textblock --img_paths image/dir/path/*.jpg --use_show --output_path layout_prediction.xml
```

処理対象をリストで指定する場合は次のように実行します。
```
python -m tools.process_textblock --list_path image_list_file.list --use_show --output_path layout_prediction.xml
```

