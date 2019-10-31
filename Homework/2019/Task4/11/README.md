# 基于机器学习和深度学习的DGA域名检测

## 数据集

- dga：https://data.netlab.360.com/feeds/dga/dga.txt
- 正常：：https://www.kaggle.com/cheedcheed/top1m

## 机器学习

- 特征集：香农熵、元音字比重、数字占比、重复字母占比、连续数字占比、2-gram、3-gram、是否为顶级域名
- 使用决策树模型进行训练

## 深度学习

- 基于bigram的logistic回归分类器
- 数据集划分出0.05用作最后的测试评估
- folds=10的k折交叉验证
- 每一次训练以128个样本为一个batch迭代10次
- 分别测试了一个正常样本和一个DGA样本