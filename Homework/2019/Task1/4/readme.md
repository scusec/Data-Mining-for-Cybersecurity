## 思路&原理
### 文本蕴涵（自然语言推理）

- 该任务旨在利用神经网络模型来推断一个文本中是否蕴涵着某些假设。输出结果可以为：蕴涵（entailment），矛盾（contradiction），中立（neutral），本质上是一个三分类问题。

### 整体思路
#### 该任务的图架构
![0.png](https://ae01.alicdn.com/kf/Hf3ba62c2b8c94075af7bc47e32518643u.jpg)

#### 导入预训练好的 Glove 词向量，将文本中的每一个词都转换为特征向量
>GloVe 数据集是Global Vectors for Word Representation的简称，GloVe是斯坦福大学的一个开源项目，其对来自语料库的汇总全局单词-单词共现 统计数据进行训练，并且得到的表示形式展示了单词向量空间的有趣线性子结构。该数据集在将文本转化为向量上表现较为良好。

#### 导入SNLI文本蕴涵数据集作为模型的训练集（split_data_into_scores)
> SNLI1.0包含570，000的人工手写英文句子对，人工标注了平衡的分类标签:蕴含entailment,矛盾，中性。  其中 gold_label作为对一个实例的最终判断，label1-5是人工标注结果。另外，原句与假设句都使用了两种解析来表示。

- 在本次任务中我们只使用到了 gold_label 字段（作为每一个sample的标签），label1-5 字段（用于给每个实例打分，训练模型时使用）以及原句与假设句本身。

- 所有的文本句子都使用 glove 词嵌入矩阵将其转换为特征向量序列，供模型输入使用（sequence2sequence）。

- 在训练模型时使用到的标签 y（期待值），实际上是人工标注标签 label1-5 转换为的一个 1x3 的评分矩阵（score_setup）。

- 最终每一个文本语句都被填充/裁剪为一个 30x50 的特征向量矩阵来表示（fit_to_size）。

#### 搭建神经网络模型结构
> 循环神经网络得益于其记忆功能使其擅长处理序列方面的问题，它能提取序列之间的特征，进而对序列输出进行预测。我们通常使用的循环神经网络模型都是单向的，它的下一刻预测输出是根据前面多个时刻的输入来共同影响。而有些时候预测可能需要由前面若干输入和后面若干输入共同决定，这样会更加准确。所以就提出了双向循环神经网络，其能关联到未来的数据。


- 本任务使用了双向 LSTM（一个向前层 + 一个向后层，每层64个神经元）+ 算术实现的全连接层。

- 创建需要的占位符,明确模型的输入与输出。每一次训练迭代的 bacth_size 为 128。输入的是将假设句与原句向量进行了简单的拼接之后得到的结果: 60x50 的矩阵。输出是一个 1x3 的矩阵，其中每一个维度可以近似代表分类为某个标签的概率。

```
hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
y = tf.placeholder(tf.float32, [N, 3], 'label')

x = tf.concat([hyp, evi], 1) # N, (Lh+Le), d
x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
x = tf.split(x, l_seq,)

```
- 搭建神经网络模型时其中向前/向后均为单层，每层有64个神经元。为了防止过拟合，减少训练用时，在每一个 lstm 层之后还使用了 dropout,强迫某些神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力。

- 除了 lstm 模型之外，我们还需要一个全连接层才能够达到分类的效果。这里作者没有直接借助 CNN 中已经封装好的全连接层（dense），而是直接回到神经网络建立的原理部分 （wx+b）来建立。

- **注意：**这里由于双向循环神经网络中向前和向后层共同连接着输出层，所以这里的`fc_weight`的形状是`[2*hidden_size,3]`,128代表着上一层（输出层）中有128个神经元，3表示最终要分类到3个标签。此外。我们在训练时也应当考虑到这个全连接层中权重的 `loss`，所以在这里使用了`tf.add_to_collection()`函数将这一层的损失添加到图中。

```
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)
lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)
lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)
rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back,  x, dtype=tf.float32)

fc_initializer = tf.random_normal_initializer(stddev=0.1)
fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 3],
                            initializer = fc_initializer)
fc_bias = tf.get_variable('bias', [3])
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                     tf.nn.l2_loss(fc_weight))

classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias

```
- 最后使用了梯度下降算法来收敛损失函数。

## 改进（主要是针对准确率进行改善）

- 对于词向量进行缩放(scale)。
- 尝试通过Embedding层学习词嵌入或尝试Word2vec等词嵌入模型。由于语境的不同，不同的词嵌入模型可能会对词嵌入的结果造成影响。
- 在训练以及测试时，去掉与句子语义表达无关的词汇，如：is, are等等。由于LSTM神经网络对输入数据长度有着严格的限制，对去除掉与语义表达无关的词汇后，输入中会包含更多与语义相关的信息。
- 增加 LSTM 层数，神经元的数量。增加LSTM的层数可以增加神经网络的复杂程度，但同时也会增加数据集的训练难度。
- 替换其他类型的神经网络模型如：transformer等。
- 将最后的全连接层替换为 CRF 条件随机场来约束标签特征。由于神经网络结构对数据的依赖很大，数据量的大小和质量也会严重影响模型训练的效果，故而出现了将现有的线性统计模型与神经网络结构相结合的方法，效果较好的有LSTM与CRF的结合。简单来说就是在输出端将softmax 与 CRF 结合起来，使用LSTM解决提取序列特征的问题，使用CRF有效利用了句子级别的标记信息。
