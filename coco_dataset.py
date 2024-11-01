import os
from pathlib import Path
from typing import Any, Tuple, Union
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch
import random
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes
import xml.etree.ElementTree as ET
from augmentation.reversible_augmentation import get_reversible_augmentation
import torchvision.transforms.v2.functional as F
from torchvision.ops import boxes as box_ops


class VOCDataset(Dataset):
    def __init__(self, root: Union[str, Path], ann_folder: str, classes):
        files = os.listdir(root)
        names = [os.path.splitext(file)[0] for file in files]
        self.images = [os.path.join(root, x + ".jpg") for x in names]
        self.targets = [os.path.join(ann_folder, x + ".xml") for x in names]
        self.classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        target = self.get_target(idx)
        return img, target

    def get_image(self, idx):
        return read_image(self.images[idx])

    def get_target(self, idx):
        boxes = []
        labels = []
        iscrowd = []
        anno = ET.parse(self.targets[idx]).getroot()
        size = anno.find("size")
        size = (int(size.find("height").text), int(size.find("width").text))
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text)
            iscrowd.append(difficult)
            class_name = obj.find("name").text
            labels.append(self.classes.index(class_name))
            _box = obj.find("bndbox")
            boxes.append([int(_box.find("xmin").text) - 1, int(_box.find("ymin").text) - 1,
                          int(_box.find("xmax").text) - 1, int(_box.find("ymax").text) - 1])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {
            "supervised": True,
            "image_id": idx,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=size),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        return target


class CocoDetection(Dataset):
    def __init__(self,  root: Union[str, Path], ann_file: str, classes):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.categories_to_label = {cat['id']: classes.index(cat['name']) for cat in cats}

    def get_image(self, index):
        id = self.ids[index]
        return self._load_image(id)

    def get_target(self, index):
        id = self.ids[index]
        return self._load_target(id)

    def get_file_name(self, index):
        id = self.ids[index]
        return self.coco.loadImgs(id)[0]["file_name"]

    def _load_image(self, id: int):
        path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        return read_image(path)

    def _load_target(self, id: int):
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
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class LabeledDataset(Dataset):
    def __init__(self, coco_detection: CocoDetection, transforms, sample_file):
        self.coco_detection = coco_detection
        self.transforms = transforms
        with open(sample_file, "r") as file:
            self.ids = [int(x.strip()) for x in file.readlines()]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, target = self.coco_detection[id]
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class PseudoLabelDataset(Dataset):
    def __init__(self, coco_detection: CocoDetection, sample_file, weak_transforms, strong_transforms, model, device, threshold):
        self.coco_detection = coco_detection
        self.device = device
        self.model = model
        self.threshold = threshold
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms
        self.reversible_augmentation = get_reversible_augmentation()

        with open(sample_file, "r") as file:
            self.full_ids = [int(x.strip()) for x in file.readlines()]

        self.ratio = 1
        self.ids = self.full_ids

    def get_ratio(self):
        return self.ratio

    def set_ratio(self, ratio, resample=False):
        self.ratio = min(max(0, ratio), 1)
        if resample:
            self.resample()
        else:
            self.default_sample()

    def resample(self):
        count = int(len(self.full_ids) * self.ratio)
        self.ids = random.sample(self.full_ids, count)
        print(f"Dataset resampled,{count} ({self.ratio*100:.2f}%) of unlabeled images are used for unsupervised training")

    def default_sample(self):
        count = int(len(self.full_ids) * self.ratio)
        self.ids = self.full_ids[: count]
        print(f"Dataset use default sample,{count} ({self.ratio * 100:.2f}%) of unlabeled images are used for unsupervised training")

    def __len__(self):
        return len(self.ids)

    def cover_image(self, img, preds):
        keep = preds["scores"] >= self.threshold
        remove = (preds["scores"] >= 0.7) & ~keep
        keep_box = preds["boxes"][keep].to(torch.int32)
        remove_box = preds["boxes"][remove].to(torch.int32)
        img_mask = torch.ones_like(img)

        for bbox in remove_box:
            x_min, y_min, x_max, y_max = bbox
            img_mask[:, y_min:y_max, x_min:x_max] = 0

        for bbox in keep_box:
            x_min, y_min, x_max, y_max = bbox
            img_mask[:, y_min:y_max, x_min:x_max] = 1

        return img * img_mask

    def __getitem__(self, idx):
        self.model.eval()
        img = self.weak_transforms(self.coco_detection.get_image(self.ids[idx]).to(self.device))
        _, height, width = F.get_dimensions(img)
        with torch.no_grad():
            preds = self.model([img, F.hflip(img)])
            preds[1]["boxes"][:, [0, 2]] = width - preds[1]["boxes"][:, [2, 0]]

            boxes = torch.cat([data["boxes"] for data in preds], dim=0)
            labels = torch.cat([data["labels"] for data in preds], dim=0)
            scores = torch.cat([data["scores"] for data in preds], dim=0)

            keep = scores >= self.threshold
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            keep = box_ops.batched_nms(boxes, scores, labels, 0)
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

            target = {
                "supervised": False,
                "labels": labels,
                "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width)),
                "scores": scores
            }
        img, target = self.strong_transforms(img, target)

        return img, target


class CombineDataLoader:
    class DataLoaderIter:
        def __init__(self, data_loader1, data_loader2):
            self.loader1 = iter(data_loader1)
            self.loader2 = iter(data_loader2)
            length1 = len(self.loader1)
            length2 = len(self.loader2)
            self.index = 0
            self.samples = [True]*length1 + [False]*length2
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