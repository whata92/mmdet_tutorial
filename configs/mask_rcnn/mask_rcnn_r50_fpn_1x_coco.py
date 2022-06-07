_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=1),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0
            )
        )
    )
)

dataset_type = 'COCODataset'
classes = ('ship',)
data = dict(
    train=dict(
        img_prefix='/dataset/LS-SSDD-v1.0-OPEN/JPEGImages_sub',
        classes=classes,
        ann_file='/dataset/LS-SSDD-v1.0-OPEN/train.json'),
    val=dict(
        img_prefix='/dataset/LS-SSDD-v1.0-OPEN/JPEGImages_sub',
        classes=classes,
        ann_file='/dataset/LS-SSDD-v1.0-OPEN/val.json'),
    test=dict(
        img_prefix='/dataset/LS-SSDD-v1.0-OPEN/JPEGImages_sub',
        classes=classes,
        ann_file='/dataset/LS-SSDD-v1.0-OPEN/val.json')
)
log_config = dict(  # config to register logger hook
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ]
)  # The logger used to record the training process.

load_from = '/workspace/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
work_dir = "output"
