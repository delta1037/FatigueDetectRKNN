import os
import json
import argparse
import torch

from models.classify_model import ClassifyNet
from models.model_utils import *
from utils.config import fatigue_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型训练保存路径
MODEL_TRAINING_PATH = "models/model_training/"
if not os.path.exists(MODEL_TRAINING_PATH):
    os.makedirs(MODEL_TRAINING_PATH)


if __name__ == '__main__':
    # 定义模型
    backbone_eye = ClassifyNet().to(device)
    backbone_mouth = ClassifyNet().to(device)

    # 启动训练
    epoch = 15
    while epoch >= 0:
        epoch -= 1
        print('---------------------------------------------------------------------------')

        # 读取全局模型
        print("# set local models")
        load_path = MODEL_TRAINING_PATH + "saved_model.json"
        load_model_from_args(backbone_eye, backbone_mouth, load_path)

        print("#train models")
        # 训练眼部模型
        print("------------------train eye model------------------")
        train_model(
            net_backbone=backbone_eye,
            dataset_path="data/data_local/data_eye/a/"
        )

        # 训练嘴部模型
        print("-----------------train mouth model-----------------")
        train_model(
            net_backbone=backbone_mouth,
            dataset_path="data/data_local/data_mouth/a/",
        )

        # 保存模型
        save_model_path = MODEL_TRAINING_PATH + "saved_model.json"
        eye_pth_path = MODEL_TRAINING_PATH + 'eye_model.pth'
        mouth_pth_path = MODEL_TRAINING_PATH + 'mouth_model.pth'
        save_model(
            backbone_eye, backbone_mouth,
            save_model_path,
            eye_pth_path,
            mouth_pth_path
        )

        # 转换一份ONNX模型，并创建新模型同步文件
        model_to_onnx(eye_pth_path, fatigue_config["eye_onnx_model_path"])
        model_to_onnx(mouth_pth_path, fatigue_config["mouth_onnx_model_path"])
        os.popen("touch " + fatigue_config["onnx_model_sync_file"])
