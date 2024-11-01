import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from coco_dataset import CocoDetection
DETECTION_MODEL = fasterrcnn_resnet50_fpn_v2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 6
CHECKPOINT_FREQ = 50  # set checkpoint every x epochs


TRAIN_STATE_PRINT_FREQ = 500  # print train state every x iters
EVAL_FREQ = 4000  # eval every x iters

MODEL_STORAGE = "runtime/model_storage"  # the folder for model saving

# Coco2014 Dataset
CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
           'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES = len(CLASSES)
COCO_DETECTION = CocoDetection("COCODataset2014/images", "COCODataset2014/annotations.json", CLASSES)

# SAMPLE
LABELED_SAMPLE = "COCODataset2014/samples/labeled/10%.txt"
UNLABELED_SAMPLE = "COCODataset2014/samples/unlabeled/20%.txt"
VALID_SAMPLE = "COCODataset2014/samples/valid/5%.txt"


# Unlabeled setting
UNLABELED_RANDOM_SAMPLE = False
UNLABELED_USE_RATIO = 1

# Semi Supervised Learning
SEMI_SUPERVISED_TRAIN_START = 100  # start semi-supervised learning are x epoch
PSEUDO_LABEL_THRESHOLD = 0.8
EMA_UPDATE_BETA = 0.9995
UNSUPERVISED_WEIGHT = 0.5




