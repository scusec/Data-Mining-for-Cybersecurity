# SQL注入攻击检测(SQL-Injection-Detection)

```
​本项目使用了机器学习的方法对SQL注入攻击进行检测。
```

## 数据集与模型选择
本项目主要通过收集了约5000条SQL注入语句作为正样本，以及约5000条正常SQL语句作为负样本，构成约有10000SQL语句的训练数据集。

同时，项目分别选择了 **DecisionTree 、RandomForest、Adaboost、GradientBoosting、GNB与LogisticRegression**六种机器学习算法与模型对样本数据集进行训练，最终选择综合评分最高的算法训练出的模型作为输出，并保存。

## 特征选取
本项目根据SQL注入语句的语法以及语义特点，选择了如下特征作为模型的输入，分别为：

- payload长度
- 数字字符频率
- 大写字母频率
- SQL语句信息熵
- 空格百分比
- 敏感字符占比
- 特殊字符占比

## 训练结果
​通过选取上述特征，对样本数据集进行预处理后，选择了六种不同的模型算法对SQL注入攻击进行检测。

通过多次实验，综合考量各个模型的准确率、召回率等，最终选择了**RandomForest**算法训练出的模型作为最终的预测模型选择；其训练结果如下图所示；
 
<div align=center><img src=".\image\1.png"  alt="模型训练结果"></div>
<div align=center>模型训练结果</div>

各个特征对模型影响的重要性如下图所示：

<div align=center><img src=".\image\5.png"  alt="各特征影响模型比重"></div>
<div align=center>各特征影响模型比重</div>

## 测试
- 正样本测试：输入SQL语句:" 1 AND 1=1%00 "，检测结果如下图所示：

  <div align=center><img src=".\image\3.png"  alt="正样本检测结果"></div>
  <div align=center>正样本检测结果</div>   

- 负样本测试：输入SQL语句:" t%3D1498581264 "，检测结果如下图所示：

  <div align=center><img src=".\image\2.png"  alt="负样本检测结果"></div>
  <div align=center>负样本检测结果</div>   

## 运行
1. 首先运行``sqli_detect.py``文件，该程序会自动导入数据集并对其进行预处理最终生成预测模型。
2. 然后运行``payload.py``文件，该程序自动导入已有的模型，接收用户输入的SQL语句，对其作出判断并输出相应结果。

## 程序流程图
整个程序的流程图如下所示：
<div align=center><img src=".\image\4.png"  alt="程序流程图"></div>
<div align=center>程序流程图</div>  