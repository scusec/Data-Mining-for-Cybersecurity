# Malicious Domain Detection with Ensemble Learning

### 项目背景

作为互联网核心服务的**域名系统**（domain name system，DNS），其功能是将域名和IP进行相互映射。由于在设计上**缺乏安全验证机制**, 常作为黑客发动网络攻击重要行动基础设施。例如，为逃避检测、阻断和追踪而采用Fast-Flux技术和随机域名生成算法DGA。国内外安全界已认识到分析DNS数据在发现、防范、预警外部网络威胁等方面的重要价值。因此如何快速准确地发现并阻断潜在的恶意域名及对应IP是防范未知网络攻击的重要手段和研究热点。基于这个背景，本项目旨在解决**恶意域名的分析与检测**问题。

### 数据示例

- 正常域名
  - *http://eastday.com*
  - *http://clkmon.com*
  - *http://softonic.com*
  - *http://google.com.vn*
  - *http://mailchimp.com*
- 恶意域名
  - *http://rs2pl.com*
  - *http://nelioguerson.palcomp3.com.br*
  - *http://myspace.com/fantomenk*
  - *http://vente-privee.com/vp4/Registration/Registration.aspx?*
  - *http://mistureca.blogspot.com*

### 项目特点

- 本项目可应用于对**恶意域名的识别**，在用户输入域名后，可以给出检测的结果。
- 本项目基于多个数据集最终组成了有**759836**条数据的**数据集**，其中两种样本的数据量基本达到平衡。
- 本项目基于机器学习的**集成学习**模型来构建分类器，采用了以下算法构建分类器：
  - Bagging
  - Random-Forest
  - Extra-Trees
  - AdaBoost
  - Gradient-Tree-Boosting
  - Voting-Classifier
- 经过模型训练和测试，构建的分类器的准确率、精确率、召回率、F1值均在**0.93**左右。
- 本项目还提供了**TPOT**的接口，可以自动构建机器学习管道，来进行模型构建。
---



下面以三个部分介绍本项目：

- [项目环境依赖](#env)

- [项目结构与运行方法](#env1)

- [未来的工作](#env2)

---



## <span id="env">一、项目环境依赖</span>

### 1. Anaconda集成环境
- Anaconda 4.7.10
- Python 3.5
### 2. 第三方库
- pandas 0.23.0 
- numpy 1.17.2
- tldextract 2.2.1
- scikit-learn 0.19.1
- tpot 0.10.2

## <span id="env1">二、项目结构与运行方法</span>

### 1. 项目结构

      ├── README.md                         
      ├── code                             // 存放程序代码文件
      │   ├── create_dataset.py             
      │   ├── domain_predict.py             
      │   ├── tpot_select_model.py         
      │   └── train_model.py                
      ├── data                             // 存放原始数据和数据集
      │   ├── data.csv                     
      │   └── domain.csv                   
      └── model                            // 存放训练好的模型文件
          ├── adaboost.pkl                 
          ├── bagging.pkl                  
          ├── extra_tree.pkl               
          ├── gradient_tree_boosting.pkl                
          ├── random_forest.pkl            
          ├── stdsc.pkl                    
          └── voting_classifier.pkl        

### 2. 运行方法

#### (1) 激活Anaconda环境
进入项目文件夹，并激活Anaconda环境。
```
activate sklearn
```

#### (2) 创建数据集
根据*data.csv*中的原始数据提取特征并构建名为*domain.csv*的数据集。
```
python create_dataset.py
```
*注：使用原始数据之前需要先进行解压*
#### (3) 训练模型
利用构建好的数据集训练模型，并将训练好的模型保存。
```
python train_model.py
```
#### (4) 检测域名
选择一个模型并导入进行域名的检测，系统能根据用户输入的域名返回检测的结果。
```
python domain_predict.py
```
*注：项目提供了训练好的模型，需要先进行解压*
#### 另：TPOT自动化机器学习接口

   TPOT是一种基于遗传算法优化机器学习管道（pipeline）的Python自动机器学习工具。简单来说，就是TPOT可以智能地探索数千个可能的pipeline，为数据集找到最好的pipeline，从而实现机器学习中最乏味的部分。

本项目提供了利用TPOT进行自动化机器学习的接口，用户可利用这个接口进行pipeline的构建。由于TPOT比较耗时，所以，建议直接采用利用上述方法已构建好的模型进行域名检测。

## <span id="env2">三、未来的工作</span>

本项目虽然在一定程度上能够解决恶意域名检测的问题，但是仍然存在着不足。未来我们的努力方向如下：
- 进一步采集数据，扩大原始数据的规模
- 进行更精细的特征工程，使提取的特征在算法中能发挥更好的作用
- 利用网格搜索等技术对模型的参数进行调优，是模型有更好的性能
- 加入在线学习的机制，能够根据用户输入的数据扩充自身数据集，从而不断优化模型
- 构建神经网络，比较其与传统机器学习算法的性能优劣

------






