import torchvision.transforms.v2.functional as F
import torch
from torchvision.tv_tensors import BoundingBoxes
import random


__all__ = ["mix_up"]


def pad(img, target, width: int, height: int):
    _, h, w = F.get_dimensions(img)
    delta_height = height - h
    delta_width = width - w
    left = random.randint(0, delta_width)
    top = random.randint(0, delta_height)
    right = delta_width - left
    bottom = delta_height - top
    fill = [left, top, right, bottom]
    img = F.pad(img, fill)
    target["boxes"] = F.pad(target["boxes"], fill)
    return img, target


def mix_up(data1, data2):
    img1, target1 = data1
    img2, target2 = data2
    _, height1, width1 = F.get_dimensions(img1)
    _, height2, width2 = F.get_dimensions(img2)
    height = max(height1, height2)
    width = max(width1, width2)

    img1, target1 = pad(img1, target1,width, height)
    img2, target2 = pad(img2, target2,width, height)

    img = img1 * 0.5 + img2 * 0.5
    target = {
        "labels": torch.cat([target1["labels"], target2["labels"]], dim=0),
        "boxes": BoundingBoxes(torch.cat([target1["boxes"], target2["boxes"]], dim=0), format="XYXY", canvas_size=(height, width)),

    }
    if "scores" in target1.keys():
        target["scores"] = torch.cat([target1["scores"], target2["scores"]], dim=0)

    if 'supervised' in target1.keys():
        target['supervised'] = target1['supervised']

    return img, target


def image_cover(img: torch.Tensor, scores, boxes, threshold):
    keep = scores >= threshold
    remove = ~keep
    keep_box = boxes[keep].to(torch.int32)
    remove_box = boxes[remove].to(torch.int32)
    remove_scores = scores[remove]
    img_mask = torch.ones_like(img)

    for bbox, score in zip(remove_box, remove_scores):
        x_min, y_min, x_max, y_max = bbox
        img_mask[:, y_min:y_max, x_min:x_max] *= torch.sqrt(1-score)

    for bbox in keep_box:
        x_min, y_min, x_max, y_max = bbox
        img_mask[:, y_min:y_max, x_min:x_max] = 1

    noise = torch.rand_like(img)
    return img * img_mask + noise * (1-img_mask)







