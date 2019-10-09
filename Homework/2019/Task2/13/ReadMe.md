# 检测恶意URL

## 依赖环境
**numpy**
**pandas**
**sklearn**
**tldextract**
**whois**
**urllib**
**ipaddress**
**datetime**
**matplotlib**

## 实验内容

  &emsp;采用随机森林算法，判断输入url是否为恶意url。
  &emsp;首先运行create_feature.py对url的特征进行提取，之后运行train_model.py进行训练，最后用predict_model.py进行预测。
  &emsp;data.csv中存放原始url，feature.csv存放提取出的特征，model_0.851190334626.pkl为训练出的模型。



## 实验结果
 ![avatar](3.PNG)