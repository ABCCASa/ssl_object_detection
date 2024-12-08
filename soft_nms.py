from typing import Tuple

import torch
import torchvision
from torch import Tensor


def one_compare_more_iou(box, boxes):
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    return iou


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    remain_scores, indices = torch.sort(scores, descending=True)
    remain_boxes = boxes[indices]
    keep_boxes = []
    keep_scores = []
    while remain_scores.numel() > 0:
        select_box = remain_boxes[0]
        select_score = remain_scores[0]

        remain_scores = remain_scores[1:]
        remain_boxes = remain_boxes[1:]
        if remain_scores.numel() > 0:
            ious = one_compare_more_iou(select_box, remain_boxes)
            overlap = ious > iou_threshold

            overlap_scores = remain_scores[overlap]
            overlap_boxes = remain_boxes[overlap]
            count = overlap_scores.numel() + 1
            sum_scores = torch.sum(overlap_scores) + select_score
            sum_boxes = torch.sum(overlap_boxes * overlap_scores.unsqueeze(1), dim=0) + select_box * select_score
            select_box = sum_boxes/sum_scores
            select_score = sum_scores/count

            remain_boxes = remain_boxes[~overlap]
            remain_scores = remain_scores[~overlap]

        keep_boxes.append(select_box)
        keep_scores.append(select_score)

    keep_boxes = torch.stack(keep_boxes)
    keep_scores = torch.stack(keep_scores)
    return keep_boxes, keep_scores


def batched_soft_nms(boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float):
    if scores.numel() == 0:
        return boxes, scores, labels
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    for class_id in torch.unique(labels):
        mask = labels == class_id
        result_boxes, result_scores = nms(boxes[mask], scores[mask], iou_threshold)
        result_labels = torch.full_like(result_scores, class_id, dtype=torch.int64)
        keep_boxes.append(result_boxes)
        keep_scores.append(result_scores)
        keep_labels.append(result_labels)
    return torch.cat(keep_boxes), torch.cat(keep_scores), torch.cat(keep_labels)

