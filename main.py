import engine
import config
import common_utils
from torch.utils.data import DataLoader
from augmentation import data_augmentation
from coco_dataset import LabeledDataset, PseudoLabelDataset, CombineDataLoader
import coco_eval
import plot
import torch

# create and load model, optimizer and lr_scheduler
student_model = config.DETECTION_MODEL(num_classes=config.NUM_CLASSES)
student_model.to(config.DEVICE)
teacher_model = config.DETECTION_MODEL(num_classes=config.NUM_CLASSES)
teacher_model.to(config.DEVICE)
optimizer = torch.optim.SGD(student_model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.995)
model_log = engine.load(config.MODEL_STORAGE, student_model, teacher_model, optimizer, lr_scheduler)

model_log.plot_eval()

# create dataset and dataLoader
coco_detection = config.COCO_DETECTION
labeled_dataset = LabeledDataset(coco_detection, data_augmentation.get_transform_supervised(), config.LABELED_SAMPLE)
labeled_loader = DataLoader(labeled_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=common_utils.collate_fn)
valid_dataset = LabeledDataset(coco_detection, data_augmentation.get_transform_valid(), config.VALID_SAMPLE)
valid_loader = DataLoader(valid_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=common_utils.collate_fn)

# ssl data
unlabeled_dataset = PseudoLabelDataset(coco_detection, config.UNLABELED_SAMPLE,data_augmentation. get_transform_unsupervised_weak(),
                                       data_augmentation.get_transform_unsupervised_strong(), teacher_model, config.DEVICE,
                                       config.PSEUDO_LABEL_THRESHOLD)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=common_utils.collate_fn)
ssl_train_loader = CombineDataLoader(labeled_loader, unlabeled_loader)


plot.plot_dataset(unlabeled_dataset, "runtime/cover", config.CLASSES)

# start training the model
target_epoch = 0
while True:
    if model_log.epoch_num >= target_epoch:
        target_epoch = common_utils.input_int(f"current {model_log.epoch_num} epoch，please enter target epoch：", 0)
        if model_log.epoch_num >= target_epoch:
            break

    # train one epoch model
    if model_log.epoch_num < config.SEMI_SUPERVISED_TRAIN_START:
        engine.full_supervised_train_one_epoch(student_model, labeled_loader, valid_loader, optimizer,
                                               lr_scheduler, model_log, config.GRADIENT_ACCUMULATION)

    else:
        if not model_log.get_ssl_init():
            model_log.set_ssl_init()
            teacher_model.load_state_dict(student_model.state_dict())
        engine.semi_supervised_train_one_epoch(student_model, teacher_model, ssl_train_loader, valid_loader, optimizer, lr_scheduler,
                                               model_log, config.EMA_UPDATE_BETA, config.UNSUPERVISED_WEIGHT, config.GRADIENT_ACCUMULATION)

    engine.save(student_model, teacher_model, optimizer, lr_scheduler, model_log, config.MODEL_STORAGE, config.CHECKPOINT_FREQ)

