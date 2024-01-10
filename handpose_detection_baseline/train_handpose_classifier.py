import os
import pickle
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path


def train_and_val(data, label):
    # 划分训练集测试集，随机打乱
    train_feature, test_feature, train_labels, test_labels = train_test_split(data, label, test_size=0.25)
    train_feature, train_labels = sklearn.utils.shuffle(train_feature, train_labels)
    test_feature, test_labels = sklearn.utils.shuffle(test_feature, test_labels)
    # 随机森林分类器
    knn_model = KNeighborsClassifier()
    knn_model.fit(train_feature, train_labels)

    test_pred = knn_model.predict(test_feature)
    acc = sklearn.metrics.accuracy_score(test_pred, test_labels)
    # 注意此处的准确率对全局任务而言是含有水分的，因为有些图片mediapipe没有成功提取出其骨骼点
    print(f"验证集手势分类准确率{acc}")

    # 保存一下训练好的模型文件
    with open("handpose_classifier.model", mode="wb") as fp:
        pickle.dump(knn_model, fp)


data = []
label = []
# 读取特征数据
feature_root_path = Path("./finger_feature/")
for class_idx, classes_dir in enumerate(os.listdir(feature_root_path)):
    for file in os.listdir(feature_root_path / classes_dir):
        with open(feature_root_path / classes_dir / file, mode="rb") as fp:
            data.append(pickle.load(fp))
            label.append(class_idx)

train_and_val(data, label)
