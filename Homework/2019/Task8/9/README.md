# README

## 代码结构

```
.
├── captcha_make.py
├── cnnlib
│   ├── __pycache__
│   │   └── network.cpython-36.pyc
│   ├── network.py
│   └── recognition_object.py
├── conf
│   ├── captcha_config.json
│   ├── sample_config.json
├── data_sv.py
├── model
│   └── checkpoint
├── requirements.txt
├── sample
│   └── origin
│       ├── 04oz_15748266361205142.png
│       ├── 0lqd_15748266363165839.png
│       ├── 0nmm_1574826636264998.png
│       ├── bunw_1574826636023233.png
├── test.py
└── train.py
```

## 网络结构

```
输入	 input
1	    卷积层 + 池化层 + 降采样层 + ReLU
2   	卷积层 + 池化层 + 降采样层 + ReLU
3	    卷积层 + 池化层 + 降采样层 + ReLU
4	    全连接 + 降采样层 + Relu
5	    全连接 + softmax
输出	 output
```

## 使用方法

### 安装依赖

```
pip install -r requirements.txt
```

### 配置

在conf/captcha_config.json中可对验证码生成配置进行设置

```python
{
  "root_dir": "sample/origin/",  # 验证码保存路径
  "image_suffix": "png",         # 验证码图片后缀
  "characters": "0123456789",    # 生成验证码的可选字符
  "count": 1000,                 # 生成验证码的图片数量
  "char_count": 4,               # 每张验证码图片上的字符数量
  "width": 100,                  # 图片宽度
  "height": 60                   # 图片高度
}
```

在conf/sample_config.json中可对训练参数进行配置

```python
{
  "origin_image_dir": "sample/origin/",  # 原始文件
  "new_image_dir": "sample/new_train/",  # 新的训练样本
  "train_image_dir": "sample/train/",    # 训练集
  "test_image_dir": "sample/test/",      # 测试集
  "api_image_dir": "sample/api/",        # api接收的图片储存路径
  "online_image_dir": "sample/online/",  # 从验证码url获取的图片的储存路径
  "local_image_dir": "sample/local/",    # 本地保存图片的路径
  "model_save_dir": "model/",            # 从验证码url获取的图片的储存路径
  "image_width": 100,                    # 图片宽度
  "image_height": 60,                    # 图片高度
  "max_captcha": 4,                      # 验证码字符个数
  "image_suffix": "png",                 # 图片文件后缀
  "char_set": "0123456789abcdefghijklmnopqrstuvwxyz",  # 验证码识别结果类别
  "use_labels_json_file": false,                       # 是否开启读取`labels.json`内容
  "remote_url": "http://127.0.0.1:6100/captcha/",      # 验证码远程获取地址
  "cycle_stop": 3000,                                  # 启动任务后的训练指定次数后停止
  "acc_stop": 0.99,                                    # 训练到指定准确率后停止
  "cycle_save": 500,                                   # 训练指定次数后定时保存模型
  "enable_gpu": 0,                                     # 是否开启GUP训练
  "train_batch_size": 128,                             # 训练时每次使用的图片张数，如果CPU或者GPU内存太小可以减少这个参数
  "test_batch_size": 100                               # 每批次测试时验证的图片张数，不要超过验证码集的总数
}

```

一般只需要修改训练图片个数(train_batch_size)即可，其余配置按照使用具体情况配置

### 数据集生成

```
python captcha_make.py
```

即可在sample/origin目录下生成[标签_时间戳.png]形式的验证码数据集

### 数据集划分

```
python data_sv.py
```

### 训练

```
python train.py
```

### 测试

```
python test.py
```

训练结果

### 训练集

在按照每轮投入512张图片进行训练

使用GPU训练次数30000轮之后(四小时)

![image-20191127120543021](http://www.daiwei.store/img/train.png)

### 测试集

![image-20191127120543021](http://www.daiwei.store/img/test.png)



