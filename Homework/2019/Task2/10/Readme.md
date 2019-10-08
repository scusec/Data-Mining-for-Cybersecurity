# 恶意URL识别 第10组

## 运行环境
python 3.7
juptyer notebook
sklearn
pandas
flask


## 文件结构
    |-- code
		|-- templates
			|-- pred_the_url.html //html模板文件
        |--get_features.ipynb 及 get_features.py //从数据集提取特征
        |--train_DecisionTrees.ipynb //训练特征
        |--predict_url.ipynb //预测url
    |-- data
        |-- data.csv //url数据集
        |-- features.csv //特征数据集
        |-- clf.pkl //保存训练好的模型
    |-- Readme.md
	

## Datasets
数据集共计14157条数据，
每条数据包括url、label
对恶意url标记为1，非恶意的url标记为0

## 特征选择
no of dots
presence of hyphen
len of url
presence of at
presence of double slash
no of subdir
no of subdomain
len of domain
no of queries
is IP
presence of Suspicious_TLD
presence of suspicious domain

## 代码解读

1. get_features.ipynb 及 get_features.py
从数据集中提取特征，写入features.csv

2. train_DecisionTrees.ipynb
采用多种算法训练特征，并将训练效果最好的模型保存到clf.pkl
训练集与测试集的比例为7:3

3. predict_url.ipynb
编写网页，输入url后返回uri的特征，及URL是否安全

## 准确率，精确率和召回率

准确率 = 分类正确的样本数/总样本数

精确率 = 真阳性样本数量与所有被分类为阳性的样本的数量的比值

召回率 = 为真阳性样本数量与样本集中全部阳性样本的数量的比值


## 参考链接
GitHub - surajr/URL-Classification: Machine learning to classify Malicious (Spam)/Benign URL's
https://github.com/surajr/URL-Classification

