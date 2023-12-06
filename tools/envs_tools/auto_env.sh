#!/bin/bash

# 安装配置模型运行环境
cd ~
## 创建虚拟环境
apt-get install python3-dev gcc -y
apt-get install libopencv-dev python3-opencv -y
apt-get install python3-venv -y
python3.8 -m venv ~/pyenv/blockchain_board
source ~/pyenv/blockchain_board/bin/activate
python -m pip install --upgrade pip
# 安装依赖包
pip install -r requirements.txt
pip install rknn_toolkit_lite2-1.5.0-cp38-cp38-linux_aarch64.whl
cd -