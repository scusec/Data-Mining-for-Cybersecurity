### 运行环境

- python 3.7
- jupyter notebook

### 依赖

- sklearn
- pandas
- numpy
- urllib

### 数据集

https://www.kaggle.com/antonyj453/urldataset

### 数据处理

#### label

将恶意url标记为1， 非恶意url标记为0

#### feature

##### （1）URL统计特征

1. URL长度
2. 是否有'@'
3. 数字个数
4. 大写字母个数
5. 前缀个数
6. 数字-字符转换频次

##### （2）URL启发式特征

1. 是否含有敏感词
2. 是否含有关键词
3. 是否含有错误端口
4. 主机名项数
5. 路径中是否含有域名
6. 目录中是否含有商标名

### 训练方法

- 随机森林 
- 33%数据用作测试集、其余为训练集

### 准确率

90.7% 

### 参考文献

[Malicious URL Detection using Machine Learning: A Survey](https://arxiv.org/abs/1701.07179)

### 使用方法

在弹出框内输入待检测的域名即可得到判别结果
