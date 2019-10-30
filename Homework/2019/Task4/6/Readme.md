# 基于机器学习的DGA域名检测

## 项目背景

DGA是一种生成随机数的算法，通常用来利用随机字符来生成C&C域名，从而逃避域名黑名单的检测。DGA域名即是利用DGA算法生成的域名，通常编码于恶意软件之中，具有较大的危害性。与普通域名相比，DGA域名通常具有随机性，不具有明显规律性。同时，DGA域名在长度、信息熵、辅音特征、元音特征、数字特征、N-gram特征等方面与普通域名也有较明显区别。

## 实验环境

- Anaconda3

- Python  3.6.5
- 数据集：
  - DGA：  https://github.com/andrewaeva/DGA
  - normal:   http://s3-us-west-1.amazonaws.com/umbrella-static/index.html

## 提取特征

- 信息熵
- 元音特征
- 辅音特征
- 是否为顶级域名
- 域名长度
- 数字特征
- bigram值
- trigram值

## 系统实现框图

![系统实现框图](C:\Users\lenovo1\Desktop\Data-Mining-for-Cybersecurity\Task4\6\Screen\系统实现框图.png)

## 结果分析

- 使用决策树算法时，准确率：87.6%，召回率：84.8%

  ![决策树](C:\Users\lenovo1\Desktop\Data-Mining-for-Cybersecurity\Task4\6\Screen\决策树.png)

- 使用随机森林算法时，准确率：88.5%，召回率：87.1%

  ![随机森林](C:\Users\lenovo1\Desktop\Data-Mining-for-Cybersecurity\Task4\6\Screen\随机森林.png)

## 待改进提高之处

- 在数据集过于庞大时，特征提取所需时间过长，效率很低。接下来可以寻找更加高校的特征提取算法对程序进行改进。
- 在使用决策树和随机森林进行训练时，准确率和召回率均低于90%。接下来可以继续增加提取的特征，或尝试用其他算法进行模型训练。





