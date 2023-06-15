_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    './ndl_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

num_classes = 16

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes)],
    mask_head=dict(num_classes=num_classes)))

checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=150)
