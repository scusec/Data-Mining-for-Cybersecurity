# Textual Entailment with TensorFlow（文本语义蕴含分析）
### 简介

文本蕴含指的是：给定一个前提（evidence），根据这个前提去推断假说文本（hypothesis）与前提的关系，一般分为蕴含关系（entailment）和矛盾关系（contradiction），这里与逻辑关系中的蕴含关系可以互相理解，但不尽相同。
- 蕴含关系（entailment）表示从premise中可以推断出hypothesis；
- 矛盾关系（contradiction）即hypothesis与premise矛盾。


### 蕴含示例

```
- Evidence1：Tom is a cat.
- Hypothesis1：Tom is a mouse.
```

- 显然，上述的前提并不能推出假设的结论，也就是蕴含不成立，为矛盾关系，当然这只是我们理想中的模型预测结果，实际的预测情况会受很多因素影响（例如模型的准确率、文本中的“is”、“the”、“a”等不表实际意义的连接词出现频率等等）
##### 这里给出一个positive的蕴含示例：

```
- Evidence2：If you help the needy, God will reward you.
- Hypothesis2：Giving money to the poor has good consequences.
```

- 从这一Positive例子我们可以感受到，两句话虽表实际意义的词均不同，但利用机器学习自然语言处理相关技术完成的文本蕴含判别的优越之处在于，他会对其用词进行提取、处理、判别，从而“真正”意义上做到语义的理解（needy=the poor、reward=good consequences）。

### 运行环境及使用说明
- 这里对运行环境进行简要的说明
1. Python版本：3.7.3（Anaconda），不过经过测试，程序在3.5、3.6、3.7版本均可正确运行；
2. 依赖的库（此处直接以conda命令呈现）：
```
conda install numpy matplotlib tensorflow tqdm
```
3. 其中anaconda会自动解析当前Python版本并安装对应版本的tensorflow库，若需要tensorflow的GPU加速可另行安装（本程序中tensor的模型训练相关运算不太耗时）；
4. 代码中一些兼容的问题已经优化完成，主要是encoding和一些未来版本的warning；

### 模型核心内容解读
> GloVe是斯坦福大学的一个开源项目，其对来自语料库的汇总全局单词-单词共现 统计数据进行训练，并且得到的表示形式展示了单词向量空间的有趣线性子结构。该数据集在将文本转化为向量上表现较为良好。
1. 程序开头的模块主要做数据集的导入，需要注意的是：

```
glove_zip_file = "glove.6B.zip"
glove_vectors_file = "glove.6B.300d.txt"

snli_zip_file = "snli_1.0.zip"
snli_dev_file = "snli_1.0_dev.txt"
snli_full_dataset_file = "snli_1.0_train.txt"
```
- glove_vectors_file这一变量指向glove词向量数据集中的文件名，其中可选择的数据集大小有50d、100d、200d、300d，可自行选择。
- 不同数据集大小对模型结果以及训练时间有一些影响。
- 但是经过测试，在不修改任何模型参数的情况下，几个数据集训练出模型的准确率均处于45%-55%左右，上下波动不大。
2. 文本到特征向量的转换：
```
sentence2sequence(sentence)
```
该函数做文本（句子）到shape为(N, D)特征矩阵的转换，其中N为sentence中words的数量，D为每个word的特征数量。
3. Word vector的可视化函数

```
visualize(sentence)
```
- 函数将词向量转换为图像进行呈现，每行表示一个单词，列表示矢量化单词的各个维度；
- 矢量化是根据与其他词的关系来训练的，因此表示的实际含义是不明确的；
- 计算机可以理解这种矢量语言；
- 在同一位置包含相似颜色的两个向量表示意义相似的单词。
