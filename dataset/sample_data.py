import os
import cv2
import numpy as np

SAMPLE_ROOT = "data/sample_data/"
if not os.path.exists(SAMPLE_ROOT):
    os.makedirs(SAMPLE_ROOT)


class SampleFaceData:
    def __init__(self, root_dir):
        # 根目录
        self.root_dir = SAMPLE_ROOT + root_dir
        os.makedirs(self.root_dir)

        self.eye_root = self.root_dir + "/eye"
        os.makedirs(self.eye_root)
        self.eye_open_root = self.root_dir + "/eye/open/"
        os.makedirs(self.eye_open_root)
        self.eye_close_root = self.root_dir + "/eye/close/"
        os.makedirs(self.eye_close_root)

        self.mouth_root = self.root_dir + "/mouth"
        os.makedirs(self.mouth_root)
        self.mouth_open_root = self.root_dir + "/mouth/open/"
        os.makedirs(self.mouth_open_root)
        self.mouth_close_root = self.root_dir + "/mouth/close/"
        os.makedirs(self.mouth_close_root)
        self.unique_tag = 0

    def add_data(self, eye_left_img, eye_right_img, mouth_img, eye_left_status="U", eye_right_status="U", mouth_status="U"):
        # 左侧眼睛图片
        if eye_left_img:
            cv_img = cv2.cvtColor(np.asarray(eye_left_img), cv2.COLOR_RGB2BGR)
            if eye_left_status == "O":
                cv2.imwrite(self.eye_open_root + "/eye_left_" + str(self.unique_tag) + ".png", cv_img)
            elif eye_left_status == "C":
                cv2.imwrite(self.eye_close_root + "/eye_left_" + str(self.unique_tag) + ".png", cv_img)
            else:
                cv2.imwrite(self.eye_root + "/eye_left_" + str(self.unique_tag) + ".png", cv_img)
        # 右侧眼睛图片
        if eye_right_img:
            cv_img = cv2.cvtColor(np.asarray(eye_right_img), cv2.COLOR_RGB2BGR)
            if eye_right_status == "O":
                cv2.imwrite(self.eye_open_root + "/eye_right_" + str(self.unique_tag) + ".png", cv_img)
            elif eye_right_status == "C":
                cv2.imwrite(self.eye_close_root + "/eye_right_" + str(self.unique_tag) + ".png", cv_img)
            else:
                cv2.imwrite(self.eye_root + "/eye_right_" + str(self.unique_tag) + ".png", cv_img)
        # 嘴巴图片
        if mouth_img:
            cv_img = cv2.cvtColor(np.asarray(mouth_img), cv2.COLOR_RGB2BGR)
            if mouth_status == "O":
                cv2.imwrite(self.mouth_open_root + "/mouth_" + str(self.unique_tag) + ".png", cv_img)
            elif mouth_status == "C":
                cv2.imwrite(self.mouth_close_root + "/mouth_" + str(self.unique_tag) + ".png", cv_img)
            else:
                cv2.imwrite(self.mouth_root + "/mouth_" + str(self.unique_tag) + ".png", cv_img)
        # 更新 tag
        self.unique_tag += 1
