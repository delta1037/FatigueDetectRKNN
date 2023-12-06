import _thread
import cv2
import os
import time
import json
import psutil
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, request

from gpio.gpio import GPIO
from utils.utils_log import *
from refer.refer_image import ReferImage
from dataset.sample_data import SampleFaceData

import warnings
warnings.filterwarnings("ignore")

# flask
app = Flask(__name__)

# GPIO 控制
pin_number = 98
gpio = GPIO()
gpio.set_direction(pin_number, "out")

# 采集状态记录
SAMPLE_STATUS = {
    "eye_status": "U",
    "mouth_status": "U"
}

# 硬件状态记录
HW_STATUS = {
    "CPU": "0%",
    "NPU": "0%"
}

# 视频大小
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# 字体加载
font_20 = ImageFont.truetype("./utils/HarmonyOS_Sans_SC_Regular.ttf", 20)
font_50 = ImageFont.truetype("./utils/HarmonyOS_Sans_SC_Regular.ttf", 50)

# 本地IP缓存
IP_ADDRESS = ""

# 获取网卡地址
def get_net_card():
    info = psutil.net_if_addrs()
    for k, v in info.items():
        for item in v:
            if item[0] == 2 and item[1].startswith('192.168.'):
                return item[1]


# 视频录制模块
class VideoRecoder:
    def __init__(self, video_name, fps=10, size=(FRAME_WIDTH, FRAME_HEIGHT)):
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
        self.out = cv2.VideoWriter(video_name, fourcc, fps, size)

    def append_frame(self, frame):
        # 将读取到的帧写入视频
        # log_info("append a new frame")
        self.out.write(frame)

    def end(self):
        # log_info("release capture")
        self.out.release()


# 疲劳检测算法
class FatigueDetect:
    def __init__(self, mouth_jdg_count=20, eye_jdg_count=30):
        self.left_eye_state_list = []
        self.right_eye_state_list = []
        self.mouth_state_list = []
        self.mouth_jdg_count = mouth_jdg_count
        self.eye_jdg_count = eye_jdg_count

    def add_state(self, left_eye_state, right_eye_state, mouth_state):
        self.left_eye_state_list.append(1 if left_eye_state == "C" else 0)
        self.right_eye_state_list.append(1 if right_eye_state == "C" else 0)
        self.mouth_state_list.append(1 if mouth_state == "O" else 0)

        mouth_radio = np.sum(np.array(self.mouth_state_list[-self.mouth_jdg_count:]) == 1) / (self.mouth_jdg_count if len(self.mouth_state_list) > self.mouth_jdg_count else len(self.mouth_state_list))
        # 处理结果看是否疲劳
        return mouth_radio > 0.7


# 蜂鸣器提示
def beep_set_high():
    gpio.set_value(pin_number, gpio.GPIO_HIGH)
    time.sleep(1)
    gpio.set_value(pin_number, gpio.GPIO_LOW)

# 硬件状态更新
def update_hw_status():
    while True:
        # 查看cpu使用率(取三次平均值)
        cpu = "vmstat 1 3|sed  '1d'|sed  '1d'|awk '{print $15}'"
        stdout = os.popen(cpu)
        cpu = stdout.readlines()
        HW_STATUS["CPU"] = str(round((100 - (int(cpu[0]) + int(cpu[1]) + int(cpu[2])) / 3), 2)) + '%'

        # 查看NPU使用率（无法查看）

# 图片显示添加提示信息
def image_add_tag(frame, refer_res, fatigue_status, frame_rate=0, debug=False):
    draw_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(draw_img)

    color_abnormal = (255, 0, 0)
    color_normal = (0, 255, 127)
    if debug:
        # 人脸位置信息
        draw.rectangle(((refer_res["face_box"][0], refer_res["face_box"][1]),
                        (refer_res["face_box"][0] + refer_res["face_box"][2],
                        refer_res["face_box"][1] + refer_res["face_box"][3])), fill=None,
                    outline=color_normal,
                    width=2)

        # 人脸关键点信息
        show_pfld_info = True
        if show_pfld_info:
            for i in range(98):
                radius = 1
                draw.ellipse((refer_res["face_landmarks"][i * 2] - radius, refer_res["face_landmarks"][i * 2 + 1] - radius, refer_res["face_landmarks"][i * 2] + radius, refer_res["face_landmarks"][i * 2 + 1] + radius), (0, 255, 127))

        # 眼睛、嘴巴位置（用颜色表示状态）
        draw.rectangle(((refer_res["eye_left_box"][0], refer_res["eye_left_box"][1]),
                        (refer_res["eye_left_box"][0] + refer_res["eye_left_box"][2],
                        refer_res["eye_left_box"][1] + refer_res["eye_left_box"][3])), fill=None,
                    outline=color_abnormal if refer_res["eye_left_status"] == "C" else color_normal,
                    width=2)

        draw.rectangle(((refer_res["eye_right_box"][0], refer_res["eye_right_box"][1]),
                        (refer_res["eye_right_box"][0] + refer_res["eye_right_box"][2],
                        refer_res["eye_right_box"][1] + refer_res["eye_right_box"][3])), fill=None,
                    outline=color_abnormal if refer_res["eye_right_status"] == "C" else color_normal,
                    width=2)

        draw.rectangle(((refer_res["mouth_box"][0], refer_res["mouth_box"][1]),
                        (refer_res["mouth_box"][0] + refer_res["mouth_box"][2],
                        refer_res["mouth_box"][1] + refer_res["mouth_box"][3])), fill=None,
                    outline=color_abnormal if refer_res["mouth_status"] == "O" else color_normal,
                    width=2)

    # 帧率信息
    draw.text((10, 10), "FR : " + str(frame_rate), fill=color_normal, font=font_20)
    draw.text((10, 30), "CPU : " + HW_STATUS["CPU"], fill=color_normal, font=font_20)
    draw.text((10, 50), "状态 : " + ("疲劳" if fatigue_status else "正常"), fill=(color_abnormal if fatigue_status else color_normal), font=font_20)
    if fatigue_status:
        draw.text((20, 80), "休息一下吧！", fill=color_abnormal, font=font_50)
    return cv2.cvtColor(np.asarray(draw_img), cv2.COLOR_RGB2BGR)


@app.route('/')
def index():
    return render_template('index.html')


def gen_from_stream(sample_dir="", sample_type="refer", model_type="rknn", only_show=False, video_name=""):
    # 选择采样数据
    sample_handler = None
    if sample_dir != "":
        sample_handler = SampleFaceData(sample_dir)

    # 加载推理模型
    refer_handler = ReferImage(session_type=model_type)

    # 疲劳检测
    if video_name != "":
        # 视频处理较慢，减半判断的帧数
        fatigue_detect = FatigueDetect(mouth_jdg_count = 10, eye_jdg_count=15)
    else:
        fatigue_detect = FatigueDetect()

    if video_name == "":
        # 从摄像头获取数据
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        log_info(f"frame width : {int(cap.get(3))}; frame_height : {int(cap.get(4))}")
    else:
        # 从视频文件获取数据
        cap = cv2.VideoCapture("./data/video/" + video_name)
    # 循环播放视频用
    frame_counter = 0

    # 读取数据并推理
    last_frame_t = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 视频循环播放
        frame_counter +=1
        if video_name != "" and frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 纯摄像头测试
        if only_show:
            # 模拟300ms延迟（延迟200ms+推理100ms）
            # time.sleep(0.2254)
            now_time_t = time.time()
            frame_rate = 1.0 / (now_time_t - last_frame_t)
            last_frame_t = now_time_t
            print(frame_rate)

            image = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            continue

        # 降低帧率以降低CPU使用率
        # time.sleep(0.02)

        # 保存摄像头图片
        # cv2.imwrite("_camera.png", frame)

        # 推理
        refer_ret = refer_handler.refer(frame, FRAME_WIDTH, FRAME_HEIGHT)
        # print(refer_ret)

        # 未检测到人脸
        if not refer_ret["face_detect"]:
            image = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            continue

        # log_time(f"stat_facedetect_infer: {refer_ret['stat_facedetect_infer']}")
        # log_time(f"stat_pfld_infer: {refer_ret['stat_pfld_infer']}")
        # log_time(f"stat_eye_mouth_infer: {refer_ret['stat_eye_mouth_infer']}")

        # 选择是否保存数据
        if sample_dir != "":
            if sample_type == "refer":
                # 实时推理的没有标注
                sample_handler.add_data(
                    refer_ret["eye_left_image"], refer_ret["eye_right_image"], refer_ret["mouth_image"]
                )
            else:
                # 训练数据采集时，需要标注数据
                sample_handler.add_data(
                    refer_ret["eye_left_image"], refer_ret["eye_right_image"], refer_ret["mouth_image"],
                    SAMPLE_STATUS["eye_status"], SAMPLE_STATUS["eye_status"], SAMPLE_STATUS["mouth_status"]
                )

        # 处理检测结果(使用疲劳检测算法进行状态的判断)
        fatigue_status = fatigue_detect.add_state(refer_ret["eye_left_status"], refer_ret["eye_right_status"], refer_ret["mouth_status"])

        # 采样情况下强制关闭疲劳状态
        if sample_type != "refer":
            fatigue_status = False

        # 如果有疲劳状态，设置通知
        if fatigue_status:
            _thread.start_new_thread(beep_set_high, ())

        # 在图片上显示疲劳提示信息（眼睛的框、帧率等）
        now_time_t = time.time()
        frame_rate = 1.0 / (now_time_t - last_frame_t)
        last_frame_t = now_time_t
        cv_img = image_add_tag(frame, refer_ret, fatigue_status, int(frame_rate))
        image = cv2.imencode('.jpg', cv_img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


def gen_cap_camera(capture_video=False):
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    log_info(f"frame width : {int(cap.get(3))}; frame_height : {int(cap.get(4))}")

    while True:
        # 从摄像头读取一帧数据
        ret, frame = cap.read()

        if not ret:
            break

        global recoder
        if capture_video and recoder is not None:
            recoder.append_frame(frame)

        # 前端显示视频
        image = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
    # 释放资源
    cap.release()


@app.route('/video_feed/<video_name>')
def video_feed(video_name):
    # 在服务器上模拟终端使用onnx模型推理
    refer_model = "rknn"
    if ".13." not in IP_ADDRESS:
        refer_model = "onnx"
    return Response(gen_from_stream(video_name=video_name, model_type=refer_model), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_feed')
def camera_feed():
    # time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return Response(gen_from_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/real_time_detect')
def real_time_detect():
    return render_template('real_time_detect.html')


@app.route('/user_data_sample')
def user_data_sample():
    return render_template('user_data_sample.html')


@app.route('/user_video_sample')
def user_video_sample():
    return render_template('user_video_sample.html')


recoder = None
@app.route('/sample_video_data')
def sample_video_data():
    log_info("sample video start")
    # 文件夹不存在则创建
    if not os.path.exists("./data/video"):
        os.makedirs("./data/video")
    video_name = "./data/video/video_sample-" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".mp4"

    global recoder
    if recoder is None:
        recoder = VideoRecoder(video_name)
        return Response(gen_cap_camera(True), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        log_error("video is recoding !!!")
        return "/static/default_bg.png"


@app.route('/stop_sample_video_data', methods=["POST"])
def stop_sample_video_data():
    log_info("sample video stop")
    global recoder
    if recoder is not None:
        recoder.end()
        recoder = None
    return ""


@app.route('/sample_picture_data')
def sample_picture_data():
    sample_dir = "train_data-" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return Response(gen_from_stream(sample_dir=sample_dir, sample_type="sample"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch_sample_status', methods=["POST"])
def switch_sample_status():
    # eye_status open -> close -> reopen
    # mouth_status close -> open -> close
    # 设置全局变量 SAMPLE_STATUS 的值，给摄像头采集时数据分类用
    sample_status = request.get_json()
    log_info(str(sample_status))
    if sample_status["eye_status"] == "open":
        SAMPLE_STATUS["eye_status"] = "O"
    elif sample_status["eye_status"] == "close":
        SAMPLE_STATUS["eye_status"] = "C"
    else:
        SAMPLE_STATUS["eye_status"] = "U"

    if sample_status["mouth_status"] == "open":
        log_info("mouth set open")
        SAMPLE_STATUS["mouth_status"] = "O"
    elif sample_status["mouth_status"] == "close":
        SAMPLE_STATUS["mouth_status"] = "C"
    else:
        SAMPLE_STATUS["mouth_status"] = "U"

    # 从闭眼到睁眼的状态需要给出提示
    if sample_status["eye_status"] == "reopen":
        beep_set_high()
    log_info(str(SAMPLE_STATUS))
    return "success"


if __name__ == '__main__':
    # 获取外网IP地址
    IP_ADDRESS = get_net_card()
    log_info("IP_ADDRESS : " + IP_ADDRESS)

    # 启动硬件监测现成
    _thread.start_new_thread(update_hw_status, ())
    app.run(host='0.0.0.0', port=5000, debug=False)
