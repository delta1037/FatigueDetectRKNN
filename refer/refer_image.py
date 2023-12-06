import os
import cv2
import time
import _thread
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from hide_warnings import hide_warnings

from utils.utils_log import *
from utils.config import fatigue_config
from refer.refer_utils import *
from refer.session.session_rknn import RknnSession

MODEL_ROOT = "refer/refer_models/"
RKNN_MODEL_ROOT = MODEL_ROOT + "rknn_models/"
ONNX_MODEL_ROOT = MODEL_ROOT + "onnx_models/"
RKNN_EYE_MODEL_ONLINE = RKNN_MODEL_ROOT + "eye_model_online.rknn"
RKNN_MOUTH_MODEL_ONLINE = RKNN_MODEL_ROOT + "mouth_model_online.rknn"
# 转换模型的服务器地址
RKNN_TRANS_SERVER = "http://192.168.0.11:8880/"


# 模型远程转换（板卡上安装的环境不适合做模型转换）
def trans_model():
    # 眼部模型转换
    with open(fatigue_config["eye_onnx_model_path"], 'rb') as file:
        response = requests.post(RKNN_TRANS_SERVER, files={'file': file})
    if response.ok:
        # 替换成保存返回文件的路径
        with open(RKNN_EYE_MODEL_ONLINE, 'wb') as save_file:
            save_file.write(response.content)
        print(f'File saved to {RKNN_EYE_MODEL_ONLINE}')
    else:
        print(f'Request failed with status code {response.status_code}')
        print(response.text)
        return

    # 嘴部模型转换
    with open(fatigue_config["mouth_onnx_model_path"], 'rb') as file:
        response = requests.post(RKNN_TRANS_SERVER, files={'file': file})
    if response.ok:
        # 替换成保存返回文件的路径
        with open(RKNN_MOUTH_MODEL_ONLINE, 'wb') as save_file:
            save_file.write(response.content)
        print(f'File saved to {RKNN_MOUTH_MODEL_ONLINE}')
    else:
        print(f'Request failed with status code {response.status_code}')
        print(response.text)
        return
    # 创建模型已下载标志位
    os.popen("touch " + fatigue_config["onnx_model_download"])


# 使用远程来做模型转换
def try_trans_rknn():
    # 标志位 fatigue_config["onnx_model_download"] 是进程内部的（用来做异步的文件转换）
    # 判断模型是否已下载
    if os.path.exists(fatigue_config["onnx_model_download"]):
        os.popen("rm " + fatigue_config["onnx_model_download"])
        return True

    # 标志位 fatigue_config["onnx_model_sync_file"] 是跨进程的（模型训练进程和模型推理进程）
    # 没有新的模型
    if not os.path.exists(fatigue_config["onnx_model_sync_file"]):
        return False

    # 有新的模型，启动模型转换线程
    os.popen("rm " + fatigue_config["onnx_model_sync_file"])
    _thread.start_new_thread(trans_model, ())
    return False


class ReferImage:
    def __init__(self, session_type="rknn"):
        # 推理模型类型
        self.session_type = session_type

        # 推理模型初始化
        eye_init_model = RKNN_MODEL_ROOT + "eye_model.rknn"
        mouth_init_model = RKNN_MODEL_ROOT + "mouth_model_sim.rknn"
        self.facedetect_session, self.pfld_session, self.eye_session, self.mouth_session = self.init_session(
            eye_init_model,
            mouth_init_model,
            self.session_type,
        )

        # 人脸关键点平滑处理
        self.prev_pts = []

        # 图像转换
        self.image_transform = transforms.Compose([transforms.ToTensor()])

    def init_session(self, rknn_eye_model, rknn_mouth_model, session_type="rknn"):
        if session_type == "rknn":
            facedetect_session = RknnSession(RKNN_MODEL_ROOT + "version-slim-320.rknn")
            pfld_session = RknnSession(RKNN_MODEL_ROOT + "PFLD_GhostOne_112_1_opt_sim.rknn")
            eye_session = RknnSession(rknn_eye_model)
            mouth_session = RknnSession(rknn_mouth_model)
        else:
            facedetect_session = onnxruntime.InferenceSession(ONNX_MODEL_ROOT + "version-slim-320.onnx")
            pfld_session = onnxruntime.InferenceSession(ONNX_MODEL_ROOT + "PFLD_GhostOne_112_1_opt_sim.onnx")
            eye_session = onnxruntime.InferenceSession(ONNX_MODEL_ROOT + "eye_model.onnx")
            mouth_session = onnxruntime.InferenceSession(ONNX_MODEL_ROOT + "mouth_model.onnx")
        return facedetect_session, pfld_session, eye_session, mouth_session

    @hide_warnings
    def refer(self, frame, w, h):
        # 做推理模型的更新（在线更新，原型系统训练的新模型）
        # if try_trans_rknn():
        #     self.facedetect_session, self.pfld_session, self.eye_session, self.mouth_session = self.init_session(
        #         RKNN_EYE_MODEL_ONLINE,
        #         RKNN_MOUTH_MODEL_ONLINE,
        #         self.session_type,
        #     )
        ret = {
            # 人脸
            "face_box": None,
            "face_image": None,
            "face_detect": False,
            # 人脸关键点
            "face_landmarks": None,
            # 左眼
            "eye_left_box": None,
            "eye_left_image": None,
            # 右眼
            "eye_right_box": None,
            "eye_right_image": None,
            # 嘴巴
            "mouth_box": None,
            "mouth_image": None,

            # 检测耗时信息
            "stat_facedetect_prefix": 0,
            "stat_facedetect_infer": 0,
            "stat_facedetect_suffix": 0,
            "stat_pfld_prefix": 0,
            "stat_pfld_infer": 0,
            "stat_pfld_suffix": 0,
            "stat_eye_mouth_prefix": 0,
            "stat_eye_mouth_infer": 0,
            "stat_eye_mouth_suffix": 0,

            # 检测最终状态
            "eye_left_status": "U",
            "eye_right_status": "U",
            "mouth_status": "U",
        }

        t_0 = time.time()
        # 人脸检测 图像预处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        if self.session_type == "onnx":
            # 非NPU需要转换，输入维度结构不一样
            image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # log_debug("refer input image.shape: {}".format(image.shape))

        t_1 = time.time()
        # 人脸检测 推理
        confidences, boxes = self.facedetect_session.run(None, {self.facedetect_session.get_inputs()[0].name: image})

        t_2 = time.time()
        # 人脸检测 后处理
        boxes, labels, probs = predict(confidences, boxes)
        t_3 = time.time()

        # 人脸检测 耗时
        ret["stat_facedetect_prefix"] = t_1 - t_0
        ret["stat_facedetect_infer"] = t_2 - t_1
        ret["stat_facedetect_suffix"] = t_3 - t_2

        # 这里可以做动态调整，有模型在训练时就做sleep，没有训练时就跳过sleep
        # 如果不sleepCPU占用会比较高，RKNN时CPU会到20%；ONNX时CPU会到100%
        # time.sleep(0.1)

        # 未检测到人脸就显示原视频
        # 未检测到人脸，返回
        if len(probs) == 0:
            return ret
        ret["face_detect"] = True

        t_4 = time.time()
        # 选择最大概率的框
        max_idx = 0
        max_prob = probs[0]
        for idx in range(len(probs)):
            if probs[idx] > max_prob:
                max_idx = idx
                max_prob = probs[idx]

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 人脸关键点 预处理
        ret["face_box"] = [
            boxes[max_idx][0],
            boxes[max_idx][1],
            boxes[max_idx][2] - boxes[max_idx][0],
            boxes[max_idx][3] - boxes[max_idx][1],
        ]
        ret["face_image"], scale_l, x_offset, y_offset = cut_resize_letterbox(pil_img, ret["face_box"], (112, 112))
        if self.session_type == "rknn":
            image = np.squeeze(np.array(ret["face_image"]))
            image = np.expand_dims(image, axis=0)
            inputs = {self.pfld_session.get_inputs()[0].name: image}
        else:
            pfld_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            pfld_tensor_img = get_img_tensor(ret["face_image"], False, (112, 112), pfld_transform)
            inputs = {self.pfld_session.get_inputs()[0].name: to_numpy(pfld_tensor_img)}
        t_5 = time.time()

        # 人脸关键点 推理
        outputs = self.pfld_session.run(None, inputs)

        t_6 = time.time()
        # 人脸关键点 后处理
        preds = np.squeeze(outputs)
        if len(self.prev_pts) == 0:
            # 添加上一帧的结果，增加平滑度
            for i in range(98):
                center_x = preds[i * 2] * 112 * scale_l + x_offset
                center_y = preds[i * 2 + 1] * 112 * scale_l + y_offset
                self.prev_pts.append(center_x)
                self.prev_pts.append(center_y)
        for i in range(98):
            center_x = preds[i * 2] * 112 * scale_l + x_offset
            center_y = preds[i * 2 + 1] * 112 * scale_l + y_offset

            # 平滑系数
            beta = 0.7
            smooth_center_x = center_x * beta + self.prev_pts[i * 2] * (1 - beta)
            smooth_center_y = center_y * beta + self.prev_pts[i * 2 + 1] * (1 - beta)

            # 更新上一帧
            self.prev_pts[i * 2] = smooth_center_x
            self.prev_pts[i * 2 + 1] = smooth_center_y

            # radius = 1
            # draw.ellipse((smooth_center_x - radius, smooth_center_y - radius, smooth_center_x + radius,
            #               smooth_center_y + radius), (0, 255, 127))
        ret["face_landmarks"] = self.prev_pts
        t_7 = time.time()

        # 人脸关键点 耗时
        ret["stat_pfld_prefix"] = t_5 - t_4
        ret["stat_pfld_infer"] = t_6 - t_5
        ret["stat_pfld_suffix"] = t_7 - t_6

        # 继续截取眼部和嘴巴的图片进行眼睛睁开和闭合的判别（图片大小均为64 x 64）
        # # 参考 https://wywu.github.io/projects/LAB/WFLW.html
        t_8 = time.time()
        cut_bias = 15
        ret["eye_left_box"] = [
            self.prev_pts[68 * 2] - cut_bias,
            self.prev_pts[70 * 2 + 1] - cut_bias,
            self.prev_pts[72 * 2] - self.prev_pts[68 * 2] + cut_bias * 2,
            self.prev_pts[74 * 2 + 1] - self.prev_pts[70 * 2 + 1] + cut_bias * 1.5]
        ret["eye_left_image"], _, _, _ = cut_resize_letterbox(pil_img, ret["eye_left_box"], (64, 64))

        ret["eye_right_box"] = [
            self.prev_pts[60 * 2] - cut_bias,
            self.prev_pts[62 * 2 + 1] - cut_bias,
            self.prev_pts[64 * 2] - self.prev_pts[60 * 2] + cut_bias * 2,
            self.prev_pts[66 * 2 + 1] - self.prev_pts[62 * 2 + 1] + cut_bias * 1.5]
        ret["eye_right_image"], _, _, _ = cut_resize_letterbox(pil_img, ret["eye_right_box"], (64, 64))

        ret["mouth_box"] = [
            self.prev_pts[76 * 2] - cut_bias,
            self.prev_pts[79 * 2 + 1] - cut_bias,
            self.prev_pts[82 * 2] - self.prev_pts[76 * 2] + cut_bias * 2,
            self.prev_pts[85 * 2 + 1] - self.prev_pts[79 * 2 + 1] + cut_bias * 1.5]
        ret["mouth_image"], _, _, _ = cut_resize_letterbox(pil_img, ret["mouth_box"], (64, 64))

        # 眼睛和嘴巴数据预处理
        if self.session_type == "rknn":
            cut_eye_left = get_img_tensor(ret["eye_left_image"], False, (64, 64), self.image_transform)
            cut_eye_left_inputs = {self.mouth_session.get_inputs()[0].name: to_numpy(cut_eye_left).transpose(0, 2, 3, 1)}

            cut_eye_right = get_img_tensor(ret["eye_right_image"], False, (64, 64), self.image_transform)
            cut_eye_right_inputs = {self.mouth_session.get_inputs()[0].name: to_numpy(cut_eye_right).transpose(0, 2, 3, 1)}

            # api trans
            cut_mouth = get_img_tensor(ret["mouth_image"], False, (64, 64), self.image_transform)
            cut_mouth_inputs = {self.mouth_session.get_inputs()[0].name: to_numpy(cut_mouth).transpose(0, 2, 3, 1)}

            # torch model test
            # image_transform = transforms.Compose([transforms.ToTensor()])
            # cut_mouth = np.squeeze(np.array(cut_mouth_img))
            # # # cut_mouth = Image.open(sample_mouth_data_dir + "/mouth_" + str(frame_count) + ".png").resize((64, 64))
            # # # cut_mouth = Image.open("/root/projects/blockchain_fatigue_detect/train_classify_model/test_data/mouth/open/mouth_36.png").resize((64, 64))
            # cut_mouth = image_transform(cut_mouth)
            # cut_mouth = torch.unsqueeze(cut_mouth, 0)
            # cut_mouth = to_numpy(cut_mouth)
            # print(cut_mouth.shape)
            # cut_mouth_inputs = {mouth_session.get_inputs()[0].name: cut_mouth}
        else:
            cut_eye_left = get_img_tensor(ret["eye_left_image"], False, (64, 64), self.image_transform)
            cut_eye_left_inputs = {self.eye_session.get_inputs()[0].name: to_numpy(cut_eye_left)}

            cut_eye_right = get_img_tensor(ret["eye_right_image"], False, (64, 64), self.image_transform)
            cut_eye_right_inputs = {self.eye_session.get_inputs()[0].name: to_numpy(cut_eye_right)}

            cut_mouth = get_img_tensor(ret["mouth_image"], False, (64, 64), self.image_transform)
            cut_mouth_inputs = {self.mouth_session.get_inputs()[0].name: to_numpy(cut_mouth)}

        # 推理
        t_9 = time.time()
        left_eye_output = self.eye_session.run(None, cut_eye_left_inputs)
        right_eye_output = self.eye_session.run(None, cut_eye_right_inputs)
        mouth_output = self.mouth_session.run(None, cut_mouth_inputs)
        log_debug(f"right_eye : {str(right_eye_output)}")
        log_debug(f"left_eye : {str(left_eye_output)}")
        log_debug(f"mouth : {str(mouth_output)}")

        # 后处理
        t_10 = time.time()
        right_eye_output = np.squeeze(right_eye_output)
        left_eye_output = np.squeeze(left_eye_output)
        mouth_output = np.squeeze(mouth_output)

        ret["eye_right_status"] = 'O' if np.argmax(right_eye_output, axis=0) == 0 else 'C'
        ret["eye_left_status"] = 'O' if np.argmax(left_eye_output, axis=0) == 0 else 'C'
        ret["mouth_status"] = 'O' if np.argmax(mouth_output, axis=0) == 0 else 'C'
        log_debug(f'right_eye_status : {str(ret["eye_right_status"])}')
        log_debug(f'left_eye_status : {str(ret["eye_left_status"])}')
        log_debug(f'mouth_status : {str(ret["mouth_status"])}')
        t_11 = time.time()

        # 记录眼睛和嘴巴的统计时间
        ret["stat_eye_mouth_prefix"] = t_9 - t_8
        ret["stat_eye_mouth_infer"] = t_10 - t_9
        ret["stat_eye_mouth_postfix"] = t_11 - t_10
        return ret
