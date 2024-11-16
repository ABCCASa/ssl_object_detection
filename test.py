import engine
import global_config
import common_utils
from torch.utils.data import DataLoader
from augmentation import data_augmentation
from coco_dataset import CocoDataset
import coco_eval
import global_config
import plot
from torch import nn
import torch
from augmentation.reversible_augmentation import get_reversible_augmentation
import torchvision.transforms.v2.functional as F
from torchvision.ops import boxes as box_ops
from coco_eval import evaluate
from augmentation.custom_augmentation import mix_up

# create dataset and dataLoader
coco_detection = global_config.COCO_DETECTION

valid_dataset = CocoDataset(coco_detection, data_augmentation.get_transform_valid(), global_config.VALID_SAMPLE)
valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False, collate_fn=common_utils.collate_fn)

# create and load model, optimizer and lr_scheduler
student_model = global_config.DETECTION_MODEL(num_classes=global_config.NUM_CLASSES)
student_model.to(global_config.DEVICE)


model_log = engine.load(global_config.MODEL_STORAGE, student_model)
# ssl data

img, target = mix_up(valid_dataset[0],valid_dataset[0])
plot.plot_data(img, target, global_config.CLASSES, "runtime/mix", "22.png")

img, target = mix_up(valid_dataset[7],valid_dataset[6])
plot.plot_data(img, target, global_config.CLASSES, "runtime/mix", "23.png")

img, target = mix_up(valid_dataset[11],valid_dataset[45])
plot.plot_data(img, target, global_config.CLASSES, "runtime/mix", "25.png")


class TestModel(nn.Module):
    def __init__(self, model, threshold):
        super().__init__()
        self.model = model
        self.reversible_augmentation = get_reversible_augmentation()
        self.threshold = threshold

    def forward(self, images):
        return [self.single_image(i) for i in images]

    def single_image(self, img):
        self.model.eval()
        img = F.to_dtype(img, torch.float, scale=True)
        _, _, width = F.get_dimensions(img)

        with torch.no_grad():
            preds = self.model([img, F.hflip(img)])
            preds[1]["boxes"][:, [0, 2]] = width - preds[1]["boxes"][:, [2, 0]]
            boxes = torch.cat([data["boxes"] for data in preds], dim=0)
            labels = torch.cat([data["labels"] for data in preds], dim=0)
            scores = torch.cat([data["scores"] for data in preds], dim=0)

            keep = scores >= self.threshold
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            keep = box_ops.batched_nms(boxes, scores, labels, 0.5)
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            target = {
                "labels": labels,
                "boxes": boxes,
                "scores": scores
            }
        return target

    def single_image2(self, img):
        self.model.eval()
        img = F.to_dtype(img, torch.float, scale=True)
        imgs = []
        undos = []
        for i in range(4):
            aug_img, undo = self.reversible_augmentation.apply(img)
            imgs.append(aug_img.to(global_config.DEVICE))
            undos.append(undo)

        with torch.no_grad():
            preds = self.model(imgs)
            boxes = torch.cat([self.reversible_augmentation.undo(data["boxes"], undo) for data, undo in zip(preds, undos)], dim=0).cpu()
            labels = torch.cat([data["labels"] for data in preds], dim=0).cpu()
            scores = torch.cat([data["scores"] for data in preds], dim=0).cpu()

            keep = box_ops.batched_nms(boxes, scores, labels, 0.5)
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            keep = scores >= self.threshold
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            target = {
                "labels": labels,
                "boxes": boxes,
                "scores": scores
            }
        return target


test_model = TestModel(student_model, 0)

evaluate(test_model, valid_loader, global_config.DEVICE)