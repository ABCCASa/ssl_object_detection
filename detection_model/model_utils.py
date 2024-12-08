import torch
from typing import Any, List, Optional, Tuple
from torchvision.ops import boxes as box_ops
def check_degenerate_boxes(targets):
    for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
            degen_bb: List[float] = boxes[bb_idx].tolist()
            torch._assert(
                False,
                "All bounding boxes should have positive height and width."
                f" Found invalid box {degen_bb} for target at index {target_idx}.",
            )


class TargetFusion:
    def __init__(self, iou_threshold, score_threshold, decay):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.decay = decay

    def __call__(self, history_targets, targets):
        with torch.no_grad():
            detections = []
            for history_target_per_image, target_per_image in zip(history_targets, targets):
                image_boxes = target_per_image["boxes"]
                image_labels = target_per_image["labels"]
                image_scores = target_per_image["scores"]
                fusion_count = 0
                if history_target_per_image is not None:
                    history_boxes = history_target_per_image["boxes"]
                    history_scores = history_target_per_image["scores"]
                    history_labels = history_target_per_image["labels"]
                    fusion_count = history_target_per_image["fusion_count"]
                    image_boxes = torch.cat([history_boxes, image_boxes], dim=0)
                    image_labels = torch.cat([history_labels, image_labels], dim=0)
                    image_scores = torch.cat([history_scores, image_scores], dim=0)

                    keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.iou_threshold)
                    image_boxes, image_labels, image_scores = image_boxes[keep], image_labels[keep], image_scores[keep]

                keep = image_scores >= self.score_threshold
                image_boxes, image_labels, image_scores = image_boxes[keep], image_labels[keep], image_scores[keep]
                fusion_count += 1
                detections.append(
                    {
                        "boxes": image_boxes[keep],
                        "labels": image_labels[keep],
                        "scores": image_scores[keep],
                        "fusion_count": fusion_count
                    }
                )
            return detections
