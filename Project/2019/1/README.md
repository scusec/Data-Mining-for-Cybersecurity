# 基于生成对抗网络的域名生成算法

## 文件结构

1. 数据集在`dga_data`下
2. `dga_model.py`实现了模型
3. `dga_train.py`实现了训练过程
4. `dga_reader.py`实现了数据的读取与格式化

## 依赖

tensorflow 1.14
numpy

## 程序主入口

主入口在`dga_train.py`文件下，提前创建一个空的cv文件夹来保存模型的checkpoints以及tensorboard结果