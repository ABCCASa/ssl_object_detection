from scipy.stats import ttest_ind
from scipy import stats
import numpy as np
import engine
import global_config
import os
import common_utils
import plot
from torch.utils.data import DataLoader
from augmentation import data_augmentation
from coco_dataset import CocoDataset
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
root_folder = global_config.MODEL_STORAGE
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
model_names = os.listdir(root_folder)
print("Model List:")
for i in range(len(model_names)):
    print(f"    [{i}] {model_names[i]}")
select_index = common_utils.input_int("Please select model", 0, len(model_names)-1)
model_storage_folder = os.path.join(root_folder, model_names[select_index])

student_model = fasterrcnn_resnet50_fpn_v2(num_classes=global_config.NUM_CLASSES).to(global_config.DEVICE)
model_log, train_config = engine.load(model_storage_folder, student_model)
model_log.plot_eval(index = 0)
coco_valid_set = global_config.get_coco_detection_valid()
valid_dataset = CocoDataset(coco_valid_set, data_augmentation.get_transform_valid(), train_config.VALID_SAMPLE)
valid_loader = DataLoader(valid_dataset, batch_size=global_config.EVAL_BATCH_SIZE, shuffle=False, collate_fn=common_utils.collate_fn)


key = "student"
range1 = (1.632e5, 11e5)
range2 = (3.174e5, 3.254e5)
sample1 = []
sample2 = []
index = 0

for k, v in model_log.evals.items():
    if "supervised" in v.keys():
        if range1[0] < k < range1[1]:
            sample1.append(v["supervised"][index]*100)
    else:
        if range2[0] < k < range2[1]:
            sample2.append(v[key][index]*100)



def sde(data):
    mean = np.mean(data)
    # 样本标准误差
    std_dev = np.std(data)



    print(f"Mean: {mean}, Standard Error: ±{std_dev}")



# 样本数据

sde(sample1)
sde(sample2)



# 独立样本 t 检验
t_stat, p_value = ttest_ind(sample1, sample2)
print(f"t-statistic: {t_stat}, p-value: {p_value}")