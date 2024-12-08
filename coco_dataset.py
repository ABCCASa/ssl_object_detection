import os
from pathlib import Path
from typing import Any, Tuple, Union
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch
import random
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes
import soft_nms
import global_config
import plot
import torchvision.transforms.v2.functional as F
from torchvision.ops import boxes as box_ops
from augmentation.reversible_augmentation import get_reversible_augmentation
from torch import nn
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

    def get_empty_target(self, index):
        id = self.ids[index]
        coco_annotation = self.coco.loadAnns(self.coco.getAnnIds(id))
        target = {"supervised": False,  "image_id": id, "box_count": len(coco_annotation)}
        return target

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
            target = self.coco_detection.get_empty_target(id)
            if self.transforms is not None:
                image = self.transforms(image)
            return image, target
        else:
            image, target = self.coco_detection[id]
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target


class PseudoLabelDataset(Dataset):
    def __init__(self, coco_dataset: CocoDataset, threshold, max_fuse_count=5, decay=1):
        self.coco_dataset = coco_dataset
        self.threshold = threshold
        self.history_targets = {}
        self.is_init = False
        self.max_fuse_count = max_fuse_count
        self.decay = decay

    def __len__(self):
        return len(self.coco_dataset)

    def init(self, model: nn.Module, device):
        with torch.no_grad():
            model.eval()
            model.set_ssl_mode(True)
            if self.is_init:
                return
            self.is_init = True
            for img, target in self.coco_dataset:
                img = F.to_dtype(img, torch.float, scale=True)
                id = target["image_id"]
                target = model([img.to(device)])[0]
                self.update_fusion(id, target)
            model.set_ssl_mode(False)

    def update_fusion(self, id, target):
        boxes = target["boxes"].detach().cpu()
        labels = target["labels"].detach().cpu()
        scores = target["scores"].detach().cpu()
        fuse_count = 0
        if id in self.history_targets.keys():
            history_boxes, history_labels, history_scores, fuse_count = self.history_targets[id]
            boxes = torch.cat([history_boxes, boxes], dim=0)
            labels = torch.cat([history_labels, labels], dim=0)
            scores = torch.cat([history_scores * self.decay, scores], dim=0)
            #keep = box_ops.batched_nms(boxes, scores, labels, 0.5)
            #boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        keep = scores >= self.threshold
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
        fuse_count += 1
        self.history_targets[id] = (boxes.clone(), labels.clone(), scores.clone(), fuse_count)

    def get_labels(self, id):
        boxes, labels, scores, fuse_count = self.history_targets[id]
        return boxes.clone(), labels.clone(), scores.clone(), fuse_count

    def __getitem__(self, idx):
        img, target = self.coco_dataset[idx]
        img = F.to_dtype(img, torch.float, scale=True)
        id = target["image_id"]
        boxes, labels, scores, fuse_count = self.get_labels(id)
        boxes, scores, labels = soft_nms.batched_soft_nms(boxes, scores, labels, 0.55)

        with open("runtime/log_1/log.txt", "a") as file:
            file.write(f"{fuse_count} {target['box_count']} {len(boxes)}\n")

        target["labels"] = labels
        target["boxes"] = BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["scores"] = scores
        target["fuse_count"] = fuse_count
        if idx % 1000 == 0:
            plot.plot_data(img, target, global_config.CLASSES, "runtime/ssss", f"{id}.png")
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