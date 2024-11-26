from scipy.stats import ttest_ind
from scipy import stats
import numpy as np
import engine
import global_config
import os
import common_utils
root_folder = global_config.MODEL_STORAGE
if not os.path.exists(root_folder):
    os.makedirs(root_folder)
model_names = os.listdir(root_folder)
print("Model List:")
for i in range(len(model_names)):
    print(f"    [{i}] {model_names[i]}")
select_index = common_utils.input_int("Please select model", 0, len(model_names)-1)
model_storage_folder = os.path.join(root_folder, model_names[select_index])

student_model = global_config.DETECTION_MODEL(num_classes=global_config.NUM_CLASSES)
model_log,_ = engine.load(model_storage_folder, student_model)
model_log.plot_eval()

key = "teacher"
range1 = (1.7e5, 11e5)
range2 = (2.13e5, 2.35e5)
sample1 =[]
sample2=[]
index = 0

for k, v in model_log.evals.items():
    if "supervised" in v.keys():
        if range1[0]< k <range1[1]:
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