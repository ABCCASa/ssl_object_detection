import os
from pathlib import Path
from typing import Any, Tuple, Union
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
import random
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes

import global_config
import plot
from augmentation.custom_augmentation import image_cover
import torchvision.transforms.v2.functional as F
from torchvision.ops import boxes as box_ops
from augmentation.reversible_augmentation import get_reversible_augmentation

__all__ = ["CocoDetection", "CocoDataset", "PseudoLabelDataset", "CombineDataLoader"]


class CocoDetection(Dataset):
    def __init__(self,  root: Union[str, Path], ann_file: str, classes):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.categories_to_label = {cat['id']: classes.index(cat['name']) for cat in cats}

    def get_image(self, index):
        id = self.ids[index]
        return self._load_image_with_id(id)

    def get_target(self, index):
        id = self.ids[index]
        return self._load_target_with_id(id)

    def get_file_name(self, index):
        id = self.ids[index]
        return self.coco.loadImgs(id)[0]["file_name"]

    def _load_image_with_id(self, id: int):
        path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        return read_image(path)

    def _load_target_with_id(self, id: int):
        coco_annotation = self.coco.loadAnns(self.coco.getAnnIds(id))
        img_data = self.coco.loadImgs(id)[0]
        img_size = (img_data['height'], img_data['width'])
        boxes = []
        for anno in coco_annotation:
            b = anno['bbox']
            x1, y1 = b[0], b[1]
            x2, y2 = x1+b[2], y1 + b[3]
            boxes.append([int(x1), int(y1), int(x2), int(y2)])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes = BoundingBoxes(boxes, format="XYXY", canvas_size=img_size)

        labels = [self.categories_to_label[anno['category_id']] for anno in coco_annotation]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        areas = [anno['area'] for anno in coco_annotation]
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = [anno['iscrowd'] for anno in coco_annotation]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {"supervised": True, "boxes": boxes, "labels": labels, "image_id": id, "area": areas, "iscrowd": iscrowd}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image_with_id(id)
        target = self._load_target_with_id(id)
        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoDataset(Dataset):
    def __init__(self, coco_detection: CocoDetection, transforms, sample_file=None, image_only=False):
        self.coco_detection = coco_detection
        self.transforms = transforms
        self.image_only = image_only
        if sample_file is None:
            self.ids = list(range(len(coco_detection)))
        else:
            with open(sample_file, "r") as file:
                self.ids = [int(x.strip()) for x in file.readlines()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        if self.image_only:
            image = self.coco_detection.get_image(id)
            if self.transforms is not None:
                image = self.transforms(image)
            return image, None
        else:
            image, target = self.coco_detection[id]
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target


class PseudoLabelDataset(Dataset):
    def __init__(self, coco_dataset: CocoDataset, strong_transforms, model, device, threshold, decay=0.99):
        self.coco_dataset = coco_dataset
        self.device = device
        self.model = model
        self.threshold = threshold
        self.strong_transforms = strong_transforms
        self.decay = decay

        self.reversible_augmentation = get_reversible_augmentation()
        self.history_targets = {}

        self.time = 0

    def __len__(self):
        return len(self.coco_dataset)

    def target_fusion(self, idx, boxes, labels, scores):
        if idx in self.history_targets.keys():
            history_boxes, history_labels, history_scores = self.history_targets[idx]
            history_scores *= self.decay
            keep = history_scores >= self.threshold
            history_boxes, history_labels, history_scores = history_boxes[keep], history_labels[keep], history_scores[keep]

            boxes = torch.cat([history_boxes, boxes], dim=0)
            labels = torch.cat([history_labels, labels], dim=0)
            scores = torch.cat([history_scores * self.decay, scores], dim=0)

            keep = box_ops.batched_nms(boxes, scores, labels, 0.5)
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        self.history_targets[idx] = (boxes.clone(), labels.clone(), scores.clone())
        return boxes, labels, scores

    def __getitem__(self, idx):
        self.model.eval()
        img, _ = self.coco_dataset[idx]
        img = F.to_dtype(img, torch.float, scale=True)

        with torch.no_grad():
            aug_img, undo_action = self.reversible_augmentation.apply(img)
            preds = self.model([aug_img.to(self.device)])[0]
            boxes, labels, scores = preds["boxes"].cpu(), preds["labels"].cpu(), preds["scores"].cpu()
            keep = scores >= self.threshold
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            boxes = self.reversible_augmentation.undo(boxes, undo_action)
            boxes, labels, scores = self.target_fusion(idx, boxes, labels, scores)
            target = {
                "supervised": False,
                "labels": labels,
                "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
                "scores": scores
            }

            if idx % 1000 == 0:
                plot.plot_data(img, target, global_config.CLASSES, "runtime/label",f"{idx}.png")

            img, target = self.strong_transforms(img, target)

        return img, target


class CombineDataLoader:
    class DataLoaderIter:
        def __init__(self, data_loader1, data_loader2):
            self.loader1 = iter(data_loader1)
            self.loader2 = iter(data_loader2)
            self.index = 0
            self.samples = [True]*len(self.loader1) + [False]*len(self.loader2)
            random.shuffle(self.samples)

        def __len__(self):
            return len(self.samples)

        def __next__(self):
            if self.index >= self.__len__():
                raise StopIteration

            sample = self.samples[self.index]
            self.index += 1
            if sample:
                return next(self.loader1)
            else:
                return next(self.loader2)

    def __init__(self, data_loader1, data_loader2):
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2

    def __len__(self):
        return len(self.data_loader1) + len(self.data_loader2)

    def __iter__(self):
        return self.DataLoaderIter(self.data_loader1, self.data_loader2)