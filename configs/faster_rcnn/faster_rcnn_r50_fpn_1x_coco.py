_base_ = [
    '/workspace/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/workspace/configs/_base_/datasets/coco_detection.py',
    '/workspace/configs/_base_/schedules/schedule_1x.py', '/workspace/configs/_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(type='Shared2FCBBoxHead', num_classes=1)
    )
)

# Modify dataset related settings
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
        dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
    ])  # The logger used to record the training process.
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/workspace/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
work_dir = "output"
