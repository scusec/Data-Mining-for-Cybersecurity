# 基于CNN的字符型图片验证码识别

## 项目背景

验证码是网站区分正常用户访问与机器人异常访问常用的技术，并具有多种形式。字符型图片验证码是目前各大网站使用频率最高的验证码形式之一。这种类型的验证码要求用户识别给定图片上的字符，并正确输入才能通过验证。近些年来，字符型验证码逐渐有了难以辨认、输错率高的特点，使得用户难以快速通过网站验证。在本项目中，我们使用TensorFlow和CNN实现了对字符型图片验证码的自动识别，并且达到了期望的准确率。

## 实验环境

- Anaconda 3
- Python  3.7.5
- 第三方库：
  - tensorflow   2.0.0
  - captcha    #生成验证码
  - PIL    #图像处理
  - random
  - numpy
- 数据集：由于Python提供了生成验证码的第三方库captcha，因此本次实验使用的数据集为自动生成的4字符随机验证码。

## 系统实现框图

![系统实现框图](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task8/6/Screen/系统实现框图.png)

## 实验结果

- 最终训练结果，测试集的准确率：87.75%，召回率：82.15%

![测试集结果](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task8/6/Screen/测试集结果.png)

- 输入单个图片文件进行识别

![单个图片文件识别结果](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task8/6/Screen/单个图片文件识别结果.png)

## 实验改进

- 由于实验设备原因，为了减少模型训练的时间，本次实验将期望准确率调的较低。接下来可考虑采用配置更高的设备进行模型训练，已达到更加高的准确率。
- 本系统目前只能对字符型图片验证码进行识别，接下来可考虑构建出能识别更多类型验证码的模型。
