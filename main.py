import engine
import global_config
import common_utils
from torch.utils.data import DataLoader
from augmentation import data_augmentation
from coco_dataset import CocoDataset, PseudoLabelDataset, CombineDataLoader
import torch
import os
from model_log import ModelLog
from train_config import TrainConfig

# model select
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
teacher_model = global_config.DETECTION_MODEL(num_classes=global_config.NUM_CLASSES)
teacher_model.to(global_config.DEVICE)
optimizer = torch.optim.SGD(student_model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.995)
model_log, train_config = engine.load(model_storage_folder, student_model, teacher_model, optimizer, lr_scheduler)
print()
train_config.print_out()

# create dataset and dataLoader
coco_train_set = global_config.get_coco_detection_train()
coco_valid_set = global_config.get_coco_detection_valid()
labeled_dataset = CocoDataset(coco_train_set, data_augmentation.get_transform_supervised(), train_config.LABELED_SAMPLE)
labeled_loader = DataLoader(labeled_dataset, batch_size=global_config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=common_utils.collate_fn)
valid_dataset = CocoDataset(coco_valid_set, data_augmentation.get_transform_valid(), train_config.VALID_SAMPLE)
valid_loader = DataLoader(valid_dataset, batch_size=global_config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=common_utils.collate_fn)

# ssl data
unlabeled_dataset = CocoDataset(coco_train_set, None, train_config.UNLABELED_SAMPLE, image_only=True)
pseudo_label_dataset = PseudoLabelDataset(unlabeled_dataset, train_config.PSEUDO_LABEL_THRESHOLD)
pseudo_label_loader = DataLoader(pseudo_label_dataset, batch_size=global_config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=common_utils.collate_fn)
ssl_train_loader = CombineDataLoader(labeled_loader, pseudo_label_loader)


# start training the model
target_epoch = 0
while True:
    if model_log.epoch_num >= target_epoch:
        target_epoch = common_utils.input_int(f"current {model_log.epoch_num} epoch, please enter target epoch: ", 0)
        if model_log.epoch_num >= target_epoch:
            break

    # train one epoch model
    if model_log.iter_num < train_config.SEMI_SUPERVISED_TRAIN_START:
        engine.full_supervised_train_one_epoch(student_model, labeled_loader, valid_loader, optimizer, lr_scheduler, model_log, train_config)
        set_check_point = model_log.epoch_num % global_config.CHECKPOINT_FREQ == 0 or model_log.iter_num == train_config.SEMI_SUPERVISED_TRAIN_START
    else:
        if not model_log.get_ssl_init():
            model_log.set_ssl_init()
            teacher_model.load_state_dict(student_model.state_dict())
        pseudo_label_dataset.init(student_model, global_config.DEVICE)

        engine.semi_supervised_train_one_epoch(student_model, teacher_model, ssl_train_loader, valid_loader, optimizer, lr_scheduler,
                                               pseudo_label_dataset.update_fusion, model_log, train_config)

        set_check_point = model_log.epoch_num % global_config.CHECKPOINT_FREQ == 0
    engine.save(student_model, teacher_model, train_config, optimizer, lr_scheduler, model_log, model_storage_folder, set_check_point)

