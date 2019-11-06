# 基于机器学习的SQL注入检测

## SQL注入的概念

SQL语句是对数据库进行操作的一种结构化查询语句，也是网页前端在与后端数据库进行交互时采用的方式。SQL注入是发生在应用程序和数据库层的一种安全漏洞。当系统对用户输入提交的字符串的合法性检查不严格时，攻击者可以在输入的字符串中注入恶意SQL指令，当攻击者的输入被提交给web服务器时，web服务器会将注入的SQL指令当做正常指令执行，从而使数据库受到攻击。

在众多的网站安全漏洞中，SQL注入漏洞的占比高，危害大。因此开发者了解SQL注入攻击的原理，并能够设计出高效、准确地识别SQL注入语句的系统，能够大大降低网站的风险系数。

## 实验环境

- Visual Studio Code，安装插件：Python
- Python 3.7
- 数据集：https://github.com/flywangfang258/ML-for-SQL-Injection/tree/master/ML_for_SQL/data

## 文件结构

- [ ] SQL_Injection_Detection
  - [ ] .vscode                                                                    #Python解释器配置文件
    - [ ] settings.json
  - [ ] Data
    - [ ] all_data.csv                                                      #总的特征数据集
    - [ ] normal_less.csv                                              #正常语句数据集
    - [ ] sqlnew.csv                                                       #SQL注入语句数据集
    - [ ] normalMatrix.csv                                           #正常语句特征数据集
    - [ ] sqlMatrix.csv                                                   #SQL注入语句特征数据集
  - [ ] Model
    - [ ] DecisionTree.model                                      #决策树模型
    - [ ] logistic.model                                                #logistic模型
    - [ ] svm.model                                                     #svm模型
  - [ ] data.py                                                                   #数据预处理
  - [ ] feature.py                                                             #特征提取
  - [ ] DecisionTree.py                                                   #决策树算法训练
  - [ ] logistic.py                                                             #logistic算法训练
  - [ ] SVM.py                                                                 #SVM算法训练
  - [ ] testLogistic.py                                                     #测试logisitc算法训练的模型

## 实现流程

### 数据预处理与特征提取

#### 提取的特征：

- URL信息熵
- URL长度
- 最长参数长度
- 数字字符频率
- 大写字母频率
- 非法字符数量

#### 标记方法：

- 正常URL标记为1
- SQL注入语句标记为0

### 模型训练

- Decision Tree
- Logistic
- SVM

### 结果分析

- Decison Tree:  准确率：84.8%    召回率：99.5%

![DecesionTree](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task5/6/Screen/DecesionTree.png)

- Logistic：准确率：90.9%   召回率：96.6%

![Logistic](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task5/6/Screen/Logistic.png)

- SVM：准确率：89.2%   召回率：97.2%

![SVM](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task5/6/Screen/SVM.png)

- 测试Logistic模型

![testLogistic](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task5/6/Screen/testLogistic.png)



## 待改进之处

- 目前获取的数据集含有的数据量较少，接下来可以寻找更加丰富的数据集。
- 观察更多的SQL注入语句，获取SQL注入语句的更多特征。
