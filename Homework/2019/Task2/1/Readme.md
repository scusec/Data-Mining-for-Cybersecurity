# 恶意代码分析 Group 1
> 使用SVC支持向量机进行预测，最终准确率大约为93%

### 使用的数据集
- https://www.kaggle.com/antonyj453/urldataset (411k个url，其中82%good 18%bad)
- http://s3-us-west-1.amazonaws.com/umbrella-static/index.html (Cisco统计的全球点击量前1M的域名)

### 实验环境
- Python 3.7.3 (Anaconda)
- Pycharm
- 依赖库：re, csv, numpy, scikit-learn

### 特征选取及判别方法
1. 是否是IP地址组成的url (having_IP_address)；正则匹配
2. url长度 (url_length) 
3. 是否含有@符号(@ symbol)
4. url在Cisco排行榜上的排名(Cisco index)
5. 子域名的个数 (subdomain)
6. 是否含有端口 (port) 
7. 含有参数的个数 (number of paraments)
8. 是否含有锚点(#) 

### 模型选取
- 选用模型：SVC支持向量机
- Train : Test = 0.8 : 0.2
- 准确率：93%

### 其它
1. 在判断url在Cisco排行榜上的排名时，发现这411k个url 和 Cisco top1m的统计完全不重合，也许是这411k个url选取的都并非热门网站。
2. 在判断是否含有子域名时，对url字符串依据‘.’切割，但是这会“误伤”一些带IP的url，如果有域名的列表的话这部分应该会精确一些。