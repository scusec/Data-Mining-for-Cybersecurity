# Entailment with TensorFlow

​        本实验中，通过构建一个简单且快速训练的神经网络，以使用TensorFlow进行文本蕴含的实验过程！

## 简介

​       在自然语言处理(natural language processing)中，文本蕴涵(Textual entailment)是指两个文本片段有指向关系。给定一个前提文本（premise），根据这个前提去推断假说文本（hypothesis）与premise的关系，一般分为蕴含关系（entailment）、矛盾关系（contradiction）和中立关系（neutral），蕴含关系（entailment）表示从premise中可以推断出hypothesis；矛盾关系（contradiction）即hypothesis与premise矛盾，中立关系是两个句子无相关性。文本蕴含的结果就是这几个概率值，归纳为一个三分类问题。

![](https://d3ansictanv2wj.cloudfront.net/Figure_1-5c0c18591e46fb09c6f5ea7bf33c56fe.jpg)



## 文本蕴含的主要作用

​        文本蕴含在更大的应用程序中作为组件很有用。例如，问答系统可以使用文本内容来从存储的信息中验证答案。通过过滤掉不包含新信息的句子，文本内容还可以增强文档摘要。其他自然语言处理（NLP）系统也具有类似的用途。



## 环境配置

​	**Anaconda**

```
python==3.5.6
tensorflow==1.10.0
jupyter==1.0.0
notebook==5.0.0
numpy==1.15.2
```

## 主要的python库

1. tensorflow

2. numpy

3. matplotlib

4. TQDM（可选择）

5. Jupyter

### 有关Entailment with TensorFlow.ipynb的修改记录

- 环境配置，请移步[原项目](https://github.com/Steven-Hewitt/Entailment-with-Tensorflow)

- cell[5]中

  ```
  with open(glove_vectors_file, "r") as glove:
  修改为
  with open(glove_vectors_file, "r",encoding='utf-8') as glove:
  ```

## 问题分析和理解

- **使用斯坦福的GloVe单词向量化+SNLI数据集**

  ​          对于神经网络主要使用数值，我们要以某种方式将单词表示为数字（即单词向量化），GloVe为我们提供了良好的单词向量化后的材料；SNLI数据集是用来提取文本扩展。

  

- **将GloVe向量加载到内存中，并以空格分隔的格式反序列化为字典**

  

- **可视化单词向量化过程，将向量表示为图像**

  ​        在图像中，每一行代表一个单词，每一列代表矢量化单词的各个维度，使得成为计算机可以理解的向量语言，一般而言，在相同位置包含相似颜色的两个向量表示含义相似的词。

  

- **建立神经网络模型**

  ​       递归神经网络（RNN）是用于神经网络的序列学习工具，只有一层的隐藏输入值，在定义了一些初始值后，清除图形并添加一个LSTM层；DropoutWrapper（）的默认实现，用于循环图层，防止过拟合；

  ​        在LSTM内传递的信息上不能有效地使用辍学，因此，将在单词特征和最终输出中使用辍学-而是在展开的LSTM网络部分的第一层和最后一层有效地使用辍学。我们使用具有两个不同LSTM单位的双向RNN，这种形式的递归网络通过输入数据向前和向后运行。

  ​        LSTM的最终输出将传递到一组完全连接的层中，然后，我们将得到一个单一的实值得分，该得分指示每种类型的蕴藏物的强度，我们用它们来选择最终结果。

  

- **训练网络**

  ​        训练出的神经网络的准确率在50-55%左右，可以继续通过修改超参数和增加数据集大小来包括整个训练集来改进，同时训练时间也相应的增加。

  

- **关闭会话以释放系统资源**

  

## 改进

1. 添加更多的LSTM层，从而增加训练的神经网络准确率
2. 使用不同类型的RNN层，如GRUs。TensorFlow中包括GRUs的实现。
3. 使用更多数据集进行训练。
4. 去掉与句子语义无关的词语，如 the is等词汇。
5. 增加training_iterations_count的值，当其增加一倍时，准确率可以达到89%左右。

## 参考资料

Textual entailment with TensorFlow - O'Reilly Media
https://www.oreilly.com/learning/textual-entailment-with-tensorflow

