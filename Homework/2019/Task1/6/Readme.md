## neural_chinese_transliterator

#### 项目简介：

拼音转换为汉字是机器学习的一个应用。因为汉语不是字母文字，所以为了能够方便的在9键或者26键的输入法环境下使用这个几个仅有的数字进行输入。我们能够通过自然语言的机器学习来使得机器能够识别拼音字母之间的关系，并对其中的拼音词组进行分别，从而能够准确的预测用户输入拼音，同时转化为用户所期望的单词。

#### 问题描述：

例如在输入法中输入：“womenzaizheli”,我们能够输出：“我 _ 们 _ 在 _ 这 _ 里“

#### 项目的整体思路：

##### 1. 项目训练需要的数据：

- 我们需要引入[Leipzig Chinese Corpus][http://wortschatz.uni-leipzig.de/en/download]

> Leipzig Corpus是由德国莱比锡大学计算机学院语言自动处理专业师生开发的一套在线多语种词汇数据库。该系统后台有一个大型的语料库。

- 用于评估的数据：在eval/input.csv

##### 2.本次项目采用了Tacotron中的改进的CBHG模块

- CBHG模块的具体介绍可以参看：[[Tacotron: Towards End-to-End Speech Synthesis]][https://arxiv.org/abs/1703.10135]

- CBHG模块的中文简单介绍可以参看：[博文-Tacotron的笔记][https://blog.csdn.net/zongza/article/details/85627914]

> 模块使用步骤：
>
> 1. 输入序列，先经过K个1-D卷积，第K个卷积核（filter）通道为k，这些卷积核可以对当前以及上下文信息有效建模；
> 2. 卷积输出被堆叠（stack）一起，沿着时间轴最大池化（maxpooling）以增加当前信息不变性，stride取为1维持时间分辨率；
> 3. 然后输入到几个固定宽度的1-D卷积，将输出增加到起始的输入序列（参考ResNet连接方式），所有的卷积都采用Batch Normalization；
> 4. 输入多层的highway 网络，用以提取更高级别的特征；
> 5. 最后在顶部加入双向GRU，用于提取序列的上下文特征；

- 这是该模块模型结构：

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task1/6/image/1.png)

##### 3.开始数据的预处理：将Leipzig Chinese Corpus的中文、拼音的数据集进行处理

> 由于要处理文字信息，文字必须转换成可以量化的特征向量。将中文和拼音构成一定的组织结构，最后形成”id   拼音  中文“的结构，将这些数据整理成规范的数据集，方便我们后续构建和训练模型。对数据集中的中文忽略其词序和语法，句法，将其仅仅看做是一个词集合，或者说是词的一个组合，文档中每个词的出现都是独立的，而拼音字母同样是独立存在的，其只和后面对应的中文存在对应关系。

最后将处理后的数据存入data\zh.tsv中，数据的格式：

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task1/6/image/2.png)

```python
with codecs.open("data/zh.tsv", 'w', 'utf-8') as fout:
        with codecs.open("data/zho_news_2007-2009_1M-sentences.txt", 'r', 'utf-8') as fin:
            i = 1
            while 1:
                line = fin.readline()
                if not line: break
                
                try:
                    idx, sent = line.strip().split("\t")
                    sent = clean(sent)
                    if len(sent) > 0:
                        pnyns, hanzis = align(sent)
                        fout.write(u"{}\t{}\t{}\n".format(idx, pnyns, hanzis))
                except:
                    continue # it's okay as we have a pretty big corpus!
                
                if i % 10000 == 0: print(i, )
                i += 1
```

##### 4.将预处理后的数据作为模型的训练集进行训练：

> 使用了python的pickle模块进行数据的序列化。

- 模型的基本标准：

```python
 embed_size = 300 
 encoder_num_banks = 16
 num_highwaynet_blocks = 4 #卷积网络的层数
 maxlen = 50 # 拼音的最大长度
 minlen = 10 # 拼音的最小长度
 norm_type = "bn" # Either "bn", "ln", "ins", or None
 dropout_rate = 0.5
```

##### 5. 搭建神经网络学习模型：

> 卷积神经网络仿造生物的视知觉机制构建，可以进行监督学习和非监督学习。本项目采用的卷积神经网络模型为2014年牛津大学学者Karen Simonyan 和 Andrew Zisserman创建的VGG Net模型。

- VGG Net的卷积核的结构为为3×3。
- 随着每层输入volume的空间尺寸减小（conv和pool层的结果），volume的深度会随着卷积核数量的增加而增加。
- 每经过一次maxpolling层后，输出的深度翻倍。
- 在训练过程中使用比例抖动数据增强技术。
- 在每个conv层之后使用ReLU激活函数，使用批梯度下降优化算法进行训练。
- 在有条件的情况下，可以在4个Nvidia Titan Black GPU上训练两到三周,增强训练的强度。

#### 项目改进：

- 根据键盘上按键的转移概率与输入法输入输出之间的对齐关系，可以在该项目中增加一个纠错模块，实现一个带有拼写纠错能力的音汉转换系统。
- 可以引入基于规则的分词，也可以在训练中引入中文的语法、句法等规则，来改善当语料量过于庞大时网络泛化能力差的问题。
- 增强训练的强度，采用更加复杂庞大的语料来进行训练，提高卷积神经网络的泛化性能。
- 采用批量标准化使减轻计算层的工作量，提高训练的准确率。
