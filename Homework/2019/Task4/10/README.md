# DGA 检测

- ### DGA域名生成算法

  ​        DGA域名生成算法是攻击者用来生成用作域名的伪随机字符串，可以有效的避开黑名单列表的检测，其字符串序列是具有随机性的，该项目内容基于DGA域名和正常域名的数据，训练出用于检测DGA域名的模型

  

- #### 文件结构

  ```
  |-- code //相关代码
      |-- DGA detect.ipynb //中心代码
  |-- model
      |-- model.h5
      |-- results.pkl
  |-- Screen
      |-- Frame.jpg //系统框架图
      |-- run1.jpg 
      |-- run2.jpg 
      |-- test.jpg //测试截图
  |-- data.csv
  ```

  

- #### 实验环境

  - numpy

  - Keras

  - sklearn

  - pandas

    

- #### 代码理解

  - 使用深度学习LSTM层，从而不需要提取特征的步骤
  - build＿model（）：生成模型
  - run（）：用于生成模型并对其多次训练
  - build_model2（）：建立逻辑回归
  - run2（）：训练和测试逻辑回归模型
  - 打印相关信息
  - dga_preddict（）：用于域名预测的函数

- #### 实现框图

  见Frame.png
  



- #### 参考链接

  - https://www.freebuf.com/articles/network/139697.html
  - https://github.com/endgameinc/dga_predict