import os
import random
import pandas
from tqdm import tqdm

class_name_list = os.listdir("../dataset")

# 随机生成标签
random_class_names = []
for i in range(563):
    random_class_num = random.randint(0, 13)
    random_class_names.append(class_name_list[random_class_num])

excel_dataframe = pandas.read_excel("./submit_empty.xlsx", header=0)  # 读入表格，header=0指定第一行是字段行
for idx, class_name in enumerate(tqdm(random_class_names, desc="写入进展")):
    excel_dataframe.loc[idx, "预测结果"] = class_name
excel_dataframe.to_excel("./submit_example.xlsx", index=False)
