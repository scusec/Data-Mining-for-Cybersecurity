# 恶意代码分析 Group 2
> 只是课上初步写的代码，跑到了85-86%的准确率，还需要之后进一步优化
### 文件包含

1. 代码
2. 数据集
3. README

### 使用的数据集

- https://www.kaggle.com/antonyj453/urldataset

### 选用的特征
1. Length of url
2. Is IP(URL是否为IP组成的）
3. Token count
4. Average token length
5. 是否含有敏感词 0/1 字典自定义
6. Dot count

### 模型相关
- 选用模型：决策树 Decision Tree
- Train : Test = 0.9 : 0.1
- 准确率：85%

### 环境相关
- Python 3.7.3 (Anaconda)
- Jupyter notebook
- 依赖库：re, pandas, skleaern
