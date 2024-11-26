import torch
from torch import Tensor
import torchvision.transforms.v2.functional as F
import torchvision.transforms.v2 as v2
from typing import List, Optional, Tuple, Union, Dict
import random


class AugmentationBase:
    def apply(self, image: Tensor, boxes: Tensor):
        return image, boxes, None

    def undo(self, boxes: Tensor, undo_action):
        return boxes


class AugmentationGroup(AugmentationBase):
    def __init__(self, augmentations: List[AugmentationBase]):
        self.augmentations = augmentations

    def apply(self, image: Tensor, boxes: Tensor):
        undo_actions = []
        for aug in self.augmentations:
            image, boxes, undo_action = aug.apply(image, boxes)
            undo_actions.append(undo_action)
        return image, boxes, undo_actions

    def undo(self, boxes: Tensor, undo_actions: List):
        if undo_actions is not None:
            for i in range(len(self.augmentations)-1, -1, -1):
                boxes = self.augmentations[i].undo(boxes, undo_actions[i])
        return boxes


class HorizontalFlipAugmentation(AugmentationBase):
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, image: Tensor, boxes: Tensor):
        if random.random() < self.p:
            _, _, width = F.get_dimensions(image)
            image = F.hflip(image)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            return image, boxes, width
        return image, boxes, None

    def undo(self, boxes: Tensor, undo_action: Optional[int]):
        if undo_action is not None:
            width = undo_action
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return boxes


class ColorJitterAugmentation(AugmentationBase):
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = (0.875, 1.125),
        contrast: Union[float, Tuple[float, float]] = (0.5, 1.5),
        saturation: Union[float, Tuple[float, float]] = (0.5, 1.5),
        hue: Union[float, Tuple[float, float]] = (-0.05, 0.05),
        p=0.5
    ):
        self.jitter = v2.RandomPhotometricDistort(p=p, contrast=contrast, saturation=saturation, hue=hue, brightness=brightness)

    def apply(self, image: torch.Tensor, boxes: Tensor):
        image = self.jitter(image)
        return image, boxes, None


class ScaleJitterAugmentation(AugmentationBase):
    def __init__(self, min_scale: float = 0.7, max_scale: float = 1.3, interpolation=F.InterpolationMode.BILINEAR, anti_alias=True):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation = interpolation
        self.anti_alias = anti_alias

    def apply(self, image: torch.Tensor, boxes: Tensor):
        chanel, height, width = F.get_dimensions(image)
        scale = random.uniform(self.min_scale, self.max_scale)
        new_height = int(height*scale)
        new_width = int(width*scale)
        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation, antialias=self.anti_alias)
        boxes = boxes * scale
        return image, boxes, scale

    def undo(self, boxes: Tensor, undo_action: float):
        boxes /= undo_action
        return boxes


def get_reversible_augmentation():
    return AugmentationGroup(
        [
            ColorJitterAugmentation(),
            ScaleJitterAugmentation(),
            HorizontalFlipAugmentation()
        ]
    )