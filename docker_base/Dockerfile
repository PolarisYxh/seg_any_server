# docker build -t face-parsing-base .
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
#更换为阿里源
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security multiverse\n"\ > /etc/apt/sources.list
# 安装Python和pip
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip git
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
# COPY . .

RUN mkdir ~/.pip&&echo '[global]\ntimeout = 6000\nindex-url = http://pypi.douban.com/simple\ntrusted-host = pypi.douban.com EOF' > ~/.pip/pip.conf
RUN pip3 install opencv-python-headless  pycocotools matplotlib onnxruntime onnx numpy Flask requests flask_cors 
RUN pip3 install git+https://gitee.com/taitale/segment-anything.git
#python3 -m pip install  torch torchvision 安装卡死，改用指定网站下载安装
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install pytorch-ignite==0.1.0 torchsummary
# for opencv-python
#RUN apt install -y libgl1-mesa-glx libglib2.0-dev vim
#RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt clean & rm -rf /app & rm -rf ~/.cache/pip/*
ENV PYTHONPATH="/app/pytorch_hair_segmentation:${PYTHONPATH}"