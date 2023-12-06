from rknnlite.api import RKNNLite

from utils.utils_log import *
from refer.session.session_com import InputSet


class RknnSession:
    def __init__(self, model_name):
        self.model_name = model_name
        self.rknn_lite = RKNNLite()

        # load RKNN model
        log_info('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(self.model_name)
        if ret != 0:
            log_error('Load RKNN model failed')
            exit(ret)
        log_info('load model done')

        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            log_error('Init runtime environment failed')
            exit(ret)
        log_info('init runtime done')

    @staticmethod
    def get_inputs():
        return [InputSet("input_name")]

    def run(self, arg_1, img):
        img = img["input_name"]
        outputs = self.rknn_lite.inference(inputs=[img])
        return outputs