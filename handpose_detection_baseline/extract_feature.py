import cv2
import mediapipe as mp
import os
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def finger_distance_feature(hand_landmarks):
    """根据21个骨骼点坐标组织特征，该例程中计算了相邻手指指尖的相对距离"""
    # 转化为numpy矩阵存储, 骨骼点索引与hand_landmarks.jpg对应
    hand_landmarks_np = np.asarray([[joint_cor.x, joint_cor.y, joint_cor.z] for joint_cor in hand_landmarks.landmark])
    total_distance = 0
    # 使用掌根到每个手指根部骨骼点的距离和作为衡量手掌大小的量度
    for finger_idx in [1, 5, 9, 13, 17]:
        total_distance += np.linalg.norm(hand_landmarks_np[finger_idx] - hand_landmarks_np[0])
    tips_distance = []
    for finger_idx in [4,8,12,16]:
        tips_distance.append(np.linalg.norm(hand_landmarks_np[finger_idx] - hand_landmarks_np[finger_idx+4]))
    tips_distance = np.asarray(tips_distance)/total_distance
    return tips_distance


if __name__ == "__main__":
    dataset_path = Path("../dataset/")
    hand_classes = os.listdir(dataset_path)
    # 创建存储特征的文件夹
    feature_root_path = Path("./finger_feature")
    if not os.path.exists(feature_root_path):
        os.mkdir(feature_root_path)
    for hand_class in hand_classes:
        if not os.path.exists(feature_root_path / hand_class):
            os.mkdir(feature_root_path / hand_class)
    # 初始化mediapipe hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5)
    # 遍历每一张图片，提取其骨骼点坐标以组织保存特征
    for hand_class in tqdm(hand_classes, desc="总进展"):
        image_files = os.listdir(dataset_path / hand_class)
        for idx, file in enumerate(tqdm(image_files, desc=f"{hand_class}类别图片处理进展")):
            file_path = dataset_path / hand_class / file
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.flip(cv2.imread(str(file_path)), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # mediapipe hands没检测到手就下一张
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            hand_landmarks = results.multi_hand_landmarks[0]  # 数据集中每张图的只有一只手
            distance_feature = finger_distance_feature(hand_landmarks)
            with open(feature_root_path / hand_class / str(idx), mode="wb") as fp:
                pickle.dump(np.array(distance_feature), fp)  # 保存python对象
