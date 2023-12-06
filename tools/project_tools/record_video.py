import cv2

# 设置视频保存的参数
output_file = 'output.mp4'
fps = 10.0  # 帧率
frame_width = 640  # 视频宽度
frame_height = 480  # 视频高度

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头读取一帧数据
    ret, frame = cap.read()

    if not ret:
        break

    # 将读取到的帧写入视频
    out.write(frame)
# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()