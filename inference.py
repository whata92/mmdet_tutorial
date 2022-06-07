from mmdet.apis import init_detector, inference_detector

target = "12_9_21.jpg"
# Specify the path to model config and checkpoint file
config_file = '/workspace/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/workspace/work_dirs/mask_rcnn_r50_fpn_1x_coco/epoch_1.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = f'/dataset/LS-SSDD-v1.0-OPEN/JPEGImages_sub/{target}'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# or save the visualization results to image files
model.show_result(img, result, out_file=target)
