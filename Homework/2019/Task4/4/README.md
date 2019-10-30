# DGA Detector
## Introduction
DGA（域名生成算法）是一种利用随机字符来生成C&C域名，从而逃避域名黑名单检测的技术手段。

DGA的重要性：攻击者常常会使用域名将恶意程序连接至C&C服务器，从而达到操控受害者机器的目的。这些域名通常会被编码在恶意程序中，这也使得攻击者具有了很大的灵活性，他们可以轻松地更改这些域名以及IP。
## Dependencies
python 2.7

tensorflow==1.2


## Structure
### 模型
1.统计特征：

元音字母个数、唯一字母的个数、平均jarccard系数（两个集合交集与并集个数的比值）
```
def get_aeiou(domain):
    count = len(re.findall(r'[aeiou]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):
    count=len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count
```
2.字符序列：

字符转换为相应的ASCII编码

3.N-gram:
### 算法
1. 朴素贝叶斯
2. XGBoost
Gradient Boosting算法的一种
策略：

    将残差作为下一个弱分类器的训练数据，每个新的弱分类器的建立都是为了使得之前弱分类器的残差往梯度方向减少。

    将弱分类器联合起来，使用累加机制代替平均投票机制。

优点：

    a.传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。

    b.传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。

    c.xgboost在代价函数里加入了正则项，用于控制模型的复杂度。

    d.Shrinkage（缩减），相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。

3. RNN LSTM
4. 多层感知机MLP


## Result
使用朴素贝叶斯，基于2-Gram，准确率83%，召回率80%

使用朴素贝叶斯，基于统计特征模型，准确率75%，召回率71%

使用XGBoot算法，基于统计特征模型，准确率86%，召回率86%

使用多层感知机MLP，基于2-Gram，准确率95%，召回率95%

使用RNN LSTM算法，基于字符序列模型

## Future
RNN 算法训练时间过长...没等到结果

特征提取可能欠考虑

数据集不够大，如果数据集再大一点准确率会提升

在实际中，再结合DNS注册信息以及DNS解析信息，可以进一步提高准确率

数据集中，DGA域名相对较少，DGA域名与正常域名相差过于悬殊，需要增加DGA域名数量





