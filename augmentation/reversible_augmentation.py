import torch
from torch import Tensor
import torchvision.transforms.v2.functional as F
from typing import List, Optional
import random



class AugmentationBase:
    def apply(self, image: Tensor, boxes: Tensor, intensity: int = 1):
        return image, boxes, None

    def undo(self, boxes: Tensor, undo_action):
        return boxes


class AugmentationGroup(AugmentationBase):
    def __init__(self, augmentations: List[AugmentationBase]):
        self.augmentations = augmentations

    def apply(self, image: Tensor, boxes: Tensor, intensity: int = 1):
        undo_actions = []
        for aug in self.augmentations:
            image, boxes, undo_action = aug.apply(image, boxes, intensity)
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

    def apply(self, image: Tensor, boxes: Tensor, intensity: int = 1):
        if boxes is not None:
            boxes = boxes.clone()
        if random.random() < self.p:
            _, _, width = F.get_dimensions(image)
            image = F.hflip(image)
            if boxes is not None:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            return image, boxes, width
        return image, boxes, None

    def undo(self, boxes: Tensor, undo_action: Optional[int]):
        boxes = boxes.clone()
        if undo_action is not None:
            width = undo_action
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return boxes


class PaddingAugmentation(AugmentationBase):
    def __init__(self, max_fill=50):
        self.max_fill = max_fill

    def apply(self, image: Tensor, boxes: Tensor, intensity: int = 1):
        boxes = boxes.clone()
        _, h, w = F.get_dimensions(image)
        max_fill = int(self.max_fill * intensity)
        delta_height = random.randint(0, max_fill)
        delta_width = random.randint(0, max_fill)
        left = random.randint(0, delta_width)
        top = random.randint(0, delta_height)
        right = delta_width - left
        bottom = delta_height - top
        fill = [left, top, right, bottom]
        img = F.pad(image, fill)
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top
        return img, boxes, (left, top)

    def undo(self, boxes: Tensor, undo_action: Optional[int]):
        boxes = boxes.clone()
        left, top = undo_action
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        return boxes


class GaussianBlur(AugmentationBase):
    def __init__(self, p: float = 0.5, kernel_size=9):
        self.p = p
        self.kernel_size = kernel_size

    def apply(self, image: torch.Tensor, boxes: Tensor, intensity: int = 1):
        if random.random() < self.p:
            kernel_size = int((self.kernel_size-1)/2 * intensity * random.random()) * 2 + 1
            if kernel_size > 1:
                image = F.gaussian_blur_image(image, [kernel_size, kernel_size])
        return image, boxes, None


class ColorJitterAugmentation(AugmentationBase):
    def __init__(
        self,
        p: float = 0.5,
        brightness_shift: float = 0.2,
        contrast_shift: float = 0.5,
        saturation_shift: float = 0.5,
        hue_shift: float = 0.5,
    ):
        self.brightness_shift = brightness_shift
        self.contrast_shift = contrast_shift
        self.saturation_shift = saturation_shift
        self.hue_shift = hue_shift
        self.p = p

    def apply(self, image: torch.Tensor, boxes: Tensor, intensity: int = 1):
        if random.random() < self.p:
            brightness_shift = self.brightness_shift * intensity
            brightness = random.uniform(1 - brightness_shift, 1 + brightness_shift)
            image = F.adjust_brightness(image, brightness)

            contrast_shift = self.contrast_shift * intensity
            contrast = random.uniform(1 - contrast_shift, 1 + contrast_shift)
            image = F.adjust_contrast(image, contrast)

            saturation_shift = self.saturation_shift * intensity
            saturation = random.uniform(1 - saturation_shift, 1 + saturation_shift)
            image = F.adjust_saturation(image, saturation)

            hue_shift = self.hue_shift * intensity
            hue = random.uniform(-hue_shift, hue_shift)
            image = F.adjust_hue(image, hue)

        return image, boxes, None


def pepper_salt_noise(img, density):
    img = img.clone()
    _, height, width = F.get_dimensions(img)
    mask = torch.rand(height, width)
    half_density = density / 2
    img[:, mask < half_density] = 0
    img[:, mask > (1 - half_density)] = 255 if img.dtype == torch.uint8 else 1
    return img


class RandomPepperSaltNoise(AugmentationBase):
    def __init__(self, p=0.5, density=0.05):
        self.density = density
        self.p = p

    def apply(self, img: torch.Tensor, boxes: Tensor, intensity: int = 1):
        if random.random() < self.p:
            density = random.uniform(0, self.density) * intensity
            img = pepper_salt_noise(img, density)
        return img, boxes, None


def gaussian_noise(img, mean, sigma, clip):
    if not img.is_floating_point():
        raise ValueError(f"Image should be float dtype, got dtype={img.dtype}")
    noise = mean + torch.randn_like(img) * sigma
    img = img + noise
    if clip:
        img = torch.clamp(img, 0, 1)
    return img


class RandomGaussianNoise(AugmentationBase):
    def __init__(self, p=0.5, mean: float = 0.0, sigma: float = 0.05, clip: bool = True):
        self.p = p
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def apply(self, img: torch.Tensor, boxes: Tensor, intensity: int = 1):
        if random.random() < self.p:
            sigma = random.uniform(0, self.sigma) * intensity
            img = gaussian_noise(img, self.mean, sigma, self.clip)
        return img, boxes, None




class ScaleJitterAugmentation(AugmentationBase):
    def __init__(self, p=0.5, scale_shift=0.3, interpolation=F.InterpolationMode.BILINEAR, anti_alias=True):
        self.scale_shift = scale_shift
        self.interpolation = interpolation
        self.anti_alias = anti_alias
        self.p = p

    def apply(self, image: torch.Tensor, boxes: Tensor, intensity: int = 1):
        if random.random() < self.p:
            chanel, height, width = F.get_dimensions(image)
            scale_shift = self.scale_shift * intensity
            scale = random.uniform(1 - scale_shift, 1 + scale_shift)
            new_height = int(height*scale)
            new_width = int(width*scale)
            image = F.resize(image, [new_height, new_width], interpolation=self.interpolation, antialias=self.anti_alias)
            if boxes is not None:
                boxes = boxes * scale
            return image, boxes, scale
        else:
            return image, boxes, None

    def undo(self, boxes: Tensor, undo_action: float):
        if undo_action is not None:
            boxes /= undo_action
        return boxes


def get_reversible_augmentation():
    return AugmentationGroup(
        [
            ColorJitterAugmentation(p=1),
            ScaleJitterAugmentation(p=1),
            HorizontalFlipAugmentation(),
            GaussianBlur(),
            RandomPepperSaltNoise(),
            RandomGaussianNoise()
        ]
    )