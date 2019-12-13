# 网络空间安全数据挖掘技术 Task 8 CAPTCHA识别

### Group 2

- 方式：端对端识别

- 数据集：Python Captcha库生成

- Python requirements：

  ```Python
  tensorflow
  PIL
  matplotlib
  numpy
  ```

  

- 项目目录结构：

  - ./Code/：
  - ./images/：随机生成的验证码（因大小原因，这里删去了图片只留目录）
    - ./test/：划分的验证码测试集（因大小原因，这里删去了图片只留目录）
    - ./model/：训练过程中保存的模型目录（因大小原因，这里删去了模型只留目录）
    - ./CAPTCHA_script：CAPTCHA生成、网络搭建、训练的主要脚本
    - ./validate.ipynb：用于测试模型的ipython脚本
  - ./Screen/：测试结果
  - Readme.md：本说明文件





- 神经网络结构：

  - 输入层：每个输入为60*160图片矩阵，标签输入为4*10（做了one-hot）
  - 卷积层1：5*5卷积核，32神经元
  - 卷积层2：5*5卷积核，64神经元
  - 卷积层3：5*5卷积核，128神经元
  - Flatten
  - 全连接层1：1024神经元
  - 全连接层2:64神经元
  - 输出层（全连接层3）：10种输出
  - Compile：
    - 损失函数：sigmoid_交叉熵
    - 优化器：Adam，learning_rate = 0.001

- 测试结果：

  ![captcha_test1](C:\Users\Devin Wang\Desktop\CAPTCHA\Screen\captcha_test1.jpg)

  ![captcha_test2](C:\Users\Devin Wang\Desktop\CAPTCHA\Screen\captcha_test2.jpg)

  ![captcha_test3](C:\Users\Devin Wang\Desktop\CAPTCHA\Screen\captcha_test3.jpg)

  - 对于生成比较易于辨认的验证码（每个数字相对比较离散的）预测情况较好，但是对于中间0130这种1和3连到一起的情况模型的预测能力较差。
  - 准确率：95.90%

