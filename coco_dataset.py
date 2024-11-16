import os
from pathlib import Path
from typing import Any, Tuple, Union
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch
import random
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes
import xml.etree.ElementTree as ET

import global_config
import plot
from augmentation.custom_augmentation import image_cover
from augmentation.reversible_augmentation import get_reversible_augmentation
import torchvision.transforms.v2.functional as F
from torchvision.ops import boxes as box_ops

__all__ = ["CocoDetection", "CocoDataset", "PseudoLabelDataLoader", "CombineDataLoader"]


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


class PseudoLabelDataLoader(Dataset):
    def __init__(self, dataloader: DataLoader, transforms, model, device, threshold):
        self.dataloader = dataloader
        self.device = device
        self.model = model
        self.threshold = threshold
        self.transforms = transforms

    def __len__(self):
        return len(self.dataloader)

    def generate_data(self, raw_images):
        self.model.eval()
        raw_images = [img.to(self.device) for img in raw_images]
        flip_images = [F.hflip(img) for img in raw_images]
        with torch.no_grad():
            raw_preds = self.model(raw_images)
            flip_preds = self.model(flip_images)
            out_images = []
            out_targets = []
            for raw_img, raw_pred, flip_pred in zip(raw_images, raw_preds, flip_preds):
                image, target = self.post_process(raw_img, raw_pred, flip_pred)
                out_images.append(image)
                out_targets.append(target)
        return out_images, out_targets

    def post_process(self, img, raw_pred, flip_pred):
        with torch.no_grad():
            _, height, width = F.get_dimensions(img)
            flip_pred["boxes"][:, [0, 2]] = width - flip_pred["boxes"][:, [2, 0]]
            boxes = torch.cat([raw_pred["boxes"], flip_pred["boxes"]], dim=0)
            labels = torch.cat([raw_pred["labels"], flip_pred["labels"]], dim=0)
            scores = torch.cat([raw_pred["scores"], flip_pred["scores"]], dim=0)

            keep = box_ops.batched_nms(boxes, scores, labels, 0.5)
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            keep = scores >= self.threshold
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            target = {
                "supervised": False,
                "labels": labels,
                "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
                "scores": scores
            }
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target

    class DataLoaderIter:
        def __init__(self, loader):
            self.iter = iter(loader.dataloader)
            self.loader = loader

        def __len__(self):
            return len(self.iter)

        def __next__(self):
            images, _ = next(self.iter)
            return self.loader.generate_data(images)

    def __iter__(self):
        return self.DataLoaderIter(self)


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