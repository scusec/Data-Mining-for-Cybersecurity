# 恶意URL识别
## 意义
恶意链接常常出现在钓鱼网站等恶意工具中，检测恶意链接有助于我们及时发现一些攻击并及时制止。
## 文件结构
    |-- code
        |--BLSTM.ipynb //利用双向LSTM实现预测
        |--rf&dt.ipynb //利用决策树和随机森林实现预测
    |-- data
        |-- good.csv //非恶意url的数据集
        |-- bad.csv //恶意url数据集
    |-- log //训练时的日志
    |-- model//保存的模型
        |-- rf.m //随机森林模型
        |-- dt.m //决策树模型
    |-- img
## Datasets
数据集采用Kaggle的数据集以及http://www.malwaredomainlist.com 的数据集
对恶意url标记为1，非恶意的url标记为0
## 特征选择
- url中的请求参数
- url中的数字的个数
- url中非字母数字的个数
- 目录层数
- 恶意文本短语的个数
- 域名的最高一级域名是几级域名
- 请求参数的总的长度
- @符号的个数
- 连接符`_`和`-`的个数
- 含有某些后缀名的个数
- url的长度
## 算法选取
本次实验在对集中算法进行选择之后发现`决策树`和`随机森林`的效果比较好，精确度均在89%，召回率可达90%
同时我们还利用深度学习进行了识别，选取的是双向LSTM，利用keras搭建神经网络,在3轮迭代以后准确率最终可达95.23%，双向LSTM的模型可视化的图如下所示：
## 说明
训练模型已保存在/model下，测试时输入选择的模型和要测试的url即可，在notebook中调用Judge函数即可判断，`Judge(model_path,test_url)`

        In:Judge('model/dt.m','www.baidu.com')
        Out:Good url
        In:Judge('dt.m','www.crackspider.us/toolbar/install.php?pack=exe')
        Out:Bad url
        



