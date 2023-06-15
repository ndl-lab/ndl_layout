dataset_type = 'CocoDataset'
classes = ('line_main', 'line_none', 'line_inote', 'line_hnote', 'line_caption',
           'block_fig', 'block_table', 'block_pillar', 'block_folio', 'block_rubi',
           'block_chart', 'block_eqn', 'block_cfm', 'block_eng',
           'char', 'void')


data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = 768

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(image_size, image_size), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(image_size, image_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/tmp/generated/dataset_kindai_preprocessed_train.json',
        img_prefix='/tmp/dataset_kindai_preprocessed_out',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/tmp/generated/dataset_kindai_preprocessed_test.json',
        img_prefix='/tmp/dataset_kindai_preprocessed_out',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/tmp/generated/dataset_kindai_preprocessed_test.json',
        img_prefix='/tmp/dataset_kindai_preprocessed_out',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric=['bbox'], classwise=True)
