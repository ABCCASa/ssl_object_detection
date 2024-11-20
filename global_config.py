import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from coco_dataset import CocoDetection


DETECTION_MODEL = fasterrcnn_resnet50_fpn_v2

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_BATCH_SIZE = 2
ACCUMULATION_STEPS = 2


EVAL_BATCH_SIZE = 6
CHECKPOINT_FREQ = 50  # set checkpoint every x epochs


TRAIN_STATE_PRINT_FREQ = 250  # print train state every x iters
SUPERVISED_EVAL_FREQ = 5000  # eval every x iters
SEMI_SUPERVISED_EVAL_FREQ = 2000  # eval every x iters

MODEL_STORAGE = "model_storage"  # the folder for model saving

# Coco2014 Dataset
CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
           'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

NUM_CLASSES = len(CLASSES)


_COCO_TRAIN = None
_COCO_VALID = None


def get_coco_detection_train():
    global _COCO_TRAIN
    if _COCO_TRAIN is None:
        _COCO_TRAIN = CocoDetection("COCODataset/images/train2017", "COCODataset/annotations/instances_train2017.json", CLASSES)
    return _COCO_TRAIN


def get_coco_detection_valid():
    global _COCO_VALID
    if _COCO_VALID is None:
        _COCO_VALID = CocoDetection("COCODataset/images/val2017", "COCODataset/annotations/instances_val2017.json", CLASSES)
    return _COCO_VALID

