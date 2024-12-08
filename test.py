import torch

import soft_nms
from  soft_nms import batched_soft_nms
from augmentation.reversible_augmentation import get_reversible_augmentation
import global_config
import torchvision.transforms.v2 as v2
from augmentation import data_augmentation
from coco_dataset import CocoDataset
import plot
import os
import common_utils
import engine
# model select
import random


from torchvision.ops import boxes as box_ops

root_folder = global_config.MODEL_STORAGE
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
model_names = os.listdir(root_folder)
print("Model List:")
for i in range(len(model_names)):
    print(f"    [{i}] {model_names[i]}")
select_index = common_utils.input_int("Please select model, enter -1 for create a new model: ", -1, len(model_names)-1)

model_storage_folder = None
if select_index >= 0:
    model_storage_folder = os.path.join(root_folder, model_names[select_index])
else:
    model_storage_folder = os.path.join(root_folder, common_utils.get_valid_filename("Please enter new model file name: ", model_names))


# create and load model, optimizer and lr_scheduler
student_model = global_config.DETECTION_MODEL(num_classes=global_config.NUM_CLASSES)
student_model.to(global_config.DEVICE)
student_model.eval()
model_log, train_config = engine.load(model_storage_folder, student_model)


reversible_augmentation = get_reversible_augmentation()
coco_valid_set = global_config.get_coco_detection_train()

valid_dataset = CocoDataset(coco_valid_set, data_augmentation.get_transform_valid())



for data_index in [283]:
    group_boxes = []
    group_scores = []
    group_labels = []
    for i in range(15):
        with torch.no_grad():

            img, _ = valid_dataset[data_index]
            aug_img, _, undo = reversible_augmentation.apply(img, None, (10-i)/10)
            pred = student_model([aug_img.to(global_config.DEVICE)])[0]
            pred["boxes"] = reversible_augmentation.undo(pred["boxes"], undo)
            #pred["boxes"] = pred["boxes"] + 10 - torch.rand_like(pred["boxes"]) * (20)
            mask = pred["scores"] > 0.8
            pred["scores"] = pred["scores"][mask]
            pred["boxes"] = pred["boxes"][mask]
            pred["labels"] = pred["labels"][mask]
            pred["labels"][:] = 1
            group_scores.append(pred["scores"])
            group_boxes.append(pred["boxes"])
            group_labels.append(pred["labels"])
            plot.plot_data(img, pred, global_config.CLASSES, "runtime/raw", f"{data_index}_{i}.png")

            boxes, scores, labels = batched_soft_nms(torch.cat(group_boxes), torch.cat(group_scores), torch.cat(group_labels), 0.5)
            target = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                }
            plot.plot_data(img, target, global_config.CLASSES, "runtime/fusion", f"{data_index}_{i}.png" )

            boxes, scores, labels = torch.cat(group_boxes), torch.cat(group_scores), torch.cat(group_labels)
            keep = box_ops.batched_nms(boxes, scores, labels,0.5)

            target = {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep]
                }
            plot.plot_data(img, target, global_config.CLASSES, "runtime/nms", f"{data_index}_{i}.png" )