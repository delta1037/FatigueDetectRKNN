from rknn.api import RKNN
from flask import Flask, request, send_file

app = Flask(__name__)


def to_rknn(onnx_model, rknn_model):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform='rk3568')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_model)
    if ret != 0:
        print('Load model failed!')
        return 'Load model failed!'
    print('load done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return 'Build model failed!'
    print('build done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export rknn model failed!')
        return 'Export rknn model failed!'
    print('export done')
    return ''


@app.route('/onnx_to_rknn', methods=['POST'])
def onnx_to_rknn():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return 'No file found.', 400

    file = request.files['file']

    # 检查文件类型
    if file.filename == '':
        return 'No file selected.', 400
    if not allowed_file(file.filename):
        return 'Invalid file type.', 400

    # 对模型文件进行转换
    filename_prefix = "model/model-" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    onnx_model = filename_prefix + ".onnx"
    recv_file = open(onnx_model, 'w')
    recv_file.write(file.read().decode('utf-8'))
    recv_file.close()

    rknn_model = filename_prefix + ".rknn"
    ret = to_rknn(onnx_model, rknn_model)
    if ret == '':
        # 返回处理后的文件
        return send_file(rknn_model, as_attachment=True)
    else:
        return ret, 400


def allowed_file(filename):
    # 允许处理的文件类型，可以根据需求进行调整
    allowed_extensions = {'onnx'}

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)