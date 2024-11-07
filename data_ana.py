from scipy.stats import ttest_ind
from scipy import stats
import numpy as np
import engine
import global_config

student_model = global_config.DETECTION_MODEL(num_classes=global_config.NUM_CLASSES)
model_log = engine.load(global_config.MODEL_STORAGE, student_model)
model_log.plot_eval()

key = "student"
range1 = (1.6e5, 3e5)
range2 = (3.8e5, 5.3e5)
sample1 =[]
sample2=[]
index = 0

for k, v in model_log.evals.items():
    if model_log.states[k]["supervised"]:
        if range1[0]< k <range1[1]:
            sample1.append(v["supervised"][index]*100)
    else:
        if range2[0] < k < range2[1]:
            sample2.append(v[key][index]*100)



def sde(data):
    mean = np.mean(data)
    # 样本标准误差
    se = stats.sem(data)
    # 确定置信区间范围
    margin = se * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)


    #mean = np.mean(data)
    #std_error = sem(data)
    print(f"Mean: {mean}, Standard Error: ±{margin}")



# 样本数据

sde(sample1)
sde(sample2)



# 独立样本 t 检验
t_stat, p_value = ttest_ind(sample1, sample2)
print(f"t-statistic: {t_stat}, p-value: {p_value}")