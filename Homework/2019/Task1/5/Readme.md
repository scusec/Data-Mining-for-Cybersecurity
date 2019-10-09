# Introduction
Deep Text Corrector 是使用tensorflow训练的序列到序列模型，该模型能够自动纠正一些应用用词中的语法错误，如任务对话中或一些书面英语中的时态错误等。虽然目前的一些上下文敏感的贫血检查系统可以自动纠正即时消息、email或其他消息中的大量输入错误，但是它们却不能纠正简单的语法错误。比如`I am going to store`,这就是一个有着典型的语法错误的句子，合理的句子应该是`I am going to the store`,所以本项目训练神经网络来解决这一个问题。
在本项目中，我们构建序列到序列的模型，它主要纠正语法错误，并且从测试效果来看，该模型展示了较好的效果(目前效果还不怎么好)。
## Requirements
` 
  python == 3.6.8
  tensorflow == 1.13.0
  pickle == 0.7.5
  boto3 == 1.9.199
  numpy == 1.16.4
`
## Structure
code中文件结构如下：

  ```
  ├─ correct_text.py //辅助功能,这些功能可以一起训练模型,以及使用模型对错误的输入序列进行解码
  ├─ data_reader.py //一个抽象类，为能够读取源数据集并产生输入-输出对的类定义接口，其中输入是源语句的语法错误变体，而输出是原始语句。
  ├─ dtc_lambda.py //一些文件的配置
  ├─ seq2seq.py
   ├─ text_corrector_data_readers.py //包含的一些实现DataReader
  ├─ text_corrector_models.py
  ├─ TextCorrector.ipynb //集合以上代码，以交互的方式进行模型的训练和评估
  ├─ requirements.txt //需要的安装包文件，可以直接利用pip进行安装
  ├─ preprocessors //对原始数据集进行预处理
        ├─preprocess_movie_dialogs.py
  ├─ data
        ├─ dialog_corpus //对话语料库的文件夹，对每个数据的具体介绍请参加该文件夹中的readme.md
        
  ├─model //保存目前的模型，但是由于不能上传大于25M的文件，所以只能上传.index文件，如果要目前的模型文件的话戳我
        
  ```
## Some Changes
原来的代码是三年前的，现在的版本相对于之前有一些改变，具体在代码中要改什么已在下面列出
### seq2seq.py
添加
`from tensorflow.contrib.rnn.python.ops import core_rnn_cell`
`from tensorflow.contrib import rnn`
将该代码中`linear = rnn_cell._linear`修改为`linear = core_rnn_cell._linear`
该文件的代码中有concat函数，请将该函数中的前后两个参数替换
具体修改为
-   `query = array_ops.concat(query_list,1)`
-   `array_ops.concat(top_states,1)`
如果在运行时有对softmax_loss_function的报错,请用该文件中`sequence_loss_by_example`函数替换本地python安装包tensorflow目录`seq2seq.py`文件中的`sequence_loss_by_example`函数
`tf.nn.seq2seq`替换为`tf.contrib.legacy_seq2seq`
### text_corrector_models.py
修改`tf.pack` 为`tf.stack`
修改`tf.mul`为 `tf.multiply`
tf.nn.sampled_softmax_loss中将第三个参数和第四个参数的位置进行交换
### Other files
相较于之前的代码还有一些比较小的改动,在改动上述说明的之前的代码后还是有一些小报错,想不起来还有哪些,但都比较容易解决，如果不能解决,请自行google或联系邮箱fr3ya3@gmail.com

## Datasets
我们的模型的数据集来自大量语法正确的会话英语，下载的是[康奈尔电影对话语料库](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html),其中包含来自电影的会话数据超过30万行，这是目前最大的书面英语在语法上正确的数据集。

使用深度学习纠正语法错误
语言学习者会在口语和书面表达中出现各种语法错误，所以需要语法错误的自动纠正，目前语法检错的方法主要有以下几种：

-   基于规则的语法检错,虽然简单明了，但由于语言的复杂性，其覆盖面低即其召回率（Recall）较低，如果真正用在产品当中，必须投入大量人力编写规则
-   基于分类器的方法,比如最大熵分类器（Han et al., 2006）可用于检测冠词错误，其预测每一个名词短语前应该使用什么冠词（a/an/the或者不使用冠词）
-   给予模型翻译的方法，将错误的句子翻译成正确的句子。由于深度学习的发展，利用NMT(神经网络机器翻译)的方法进行语法检错也取得了不错的成果。

本模型利用训练神经网络解决这一个问题，即构建序列到序列模型，对于下载的原始数据集，需要对数据进行预处理, 运行下面的命令对原始数据集进行预处理

    python preprocessors/preprocess_movie_dialogs.py --raw_data movie_lines.txt \
                                                 --out_file preprocessed_movie_lines.txt```

已经将我们生成的训练集合和测试集放在目录`data/dialog_corpus`下
在对数据集进行预处理之后，下一步是生成在训练期间使用的输入输出对，这是通过下面的方式完成的
-   从数据集中绘制示例句子
-   随机应用某些扰动后，将输入序列设置为此句子。
-   将输出序列设置为不受干扰的句子。
例如，给出以下句子
```You must have girlfirend```
生成下面的输入输出对
```("You must have girlfirend", "You must have a girlfriend")
```
## Train
使用LSTM encoders和LSTM decoders，利用注意力机制，训练序列到序列模型，使用随机提督下降法。
### Encoder-Decoder Frame
虽然LSTM确实可以解决序列的长期依赖问题，但是对于很长的序列(>30)，LSTM的效果难以让人满意，此时需要attention mechanism。
Encoder-Decoder框架是机器翻译（Machine Translation）模型的产物，主要是在循环神经网络中应用。
在统计翻译模型中，模型的训练步骤可以分为预处理、词对齐、短语对齐、抽取短语特征、训练语言模型、学习特征权重等诸多步骤。
seq2seq模型的基本思想如下：
-   使用一个循环神经网络读取输入句子，将整个句子的信息压缩到一个固定为度的编码中(**编码器Encoder**)
-   使用另一个循环神经网络读取编码，将其“解压”为目标语言的一个句子(**解码器Decoder**)
#### 框架图
![图片alt](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task1/5/img/2.jpg)
#### Decoder
Decoder: 根据$x$的中间语义表示$c$ 和已经生成的$y_1,y_2,...,y_(i-1)$来生成$i$时刻的$y_i, y_i=g(c,y_1,y_2,...,y_(i-1))$。Decoder部分的结构和一般的语言模型基本一样：输入单词的词向量，输出为softmax层产生的单词概率，损失函数为log perplexity
#### Encoder
Encoder:对输入序列$x$进行编码，通过非线形变换转化为中间语义表示c,$c=F(x_1,x_2,...,x_m)$，它不需要softmax层
### Attention Mechanism
它对于seq2seq模型中编码器将整个句子压缩为一个固定长度的向量$c$ ，而当句子较长时其很难保存足够的语义信息，而Attention允许解码器根据当前不同的翻译内容，查阅输入句子的部分不同的单词或片段，以提高每个词或者片段的翻译精确度。

在每一步的解码过程中，将查询编码器的隐藏状态。对于整个输入序列计算每一位置（每一片段）与当前翻译内容的相关程度，即权重。再根据这个权重对各输入位置的隐藏状态进行加权平均得到“context”向量。

同时在解码下一个单词时，将context作为额外信息输入至RNN中，这样网络可以时刻读取原文中最相关的信息，而不必完全依赖于上一时刻的隐藏状态。对比社seq2seq,Attention本质上是通过加权平均，计算可变的上下文向量$c$。

注意力机制的大致原理如上所述，具体原理请参考[深度学习注意力机制](https://zhuanlan.zhihu.com/p/31547842)
### Biased Decoding
根据以下
`mask[i] == 1.0 if i in (input or corrective_tokens) else 0.0`
在下面使用

    token_probs = tf.softmax(logits)
    biased_token_probs = tf.mul(token_probs, mask)
    decoded_token = math_ops.argmax(biased_token_probs, 1)


## Results
该数据集包含来自电影脚本的304,713行，其中243,768行用于训练模型，每行、30,474行用于验证和测试集。
该模型编码器和解码器均为2层，512个隐藏单元LSTM
由于时间、电脑不带gpu、以及数据量较大，目前的迭代次数相对还有点少，只进行到step6000，但是`困惑度perplexity`已经由最初的`560.39` 下降到`1.42`,可以说明这样的训练是有一定效果的，目前的模型已经保存在文件`/model/`下,我们将会继续训练该模型，待准确率上升之后再进行更新
测试结果从notebook中的测试就可以看出有多感人(目前还未收敛，所以准确率很低很低)




### Example
处理缺少单词的句子(理想中的，目前模型还未达到该效果)

    In : decode("I went to market")
    Out: 'I went to the market'`
    In : decode("I want to have dog")
    Out: 'I want to have a dog'

## Run

### Training

    python correct_text.py --train_path /movie_dialog_train.txt \
                       --val_path /movie_dialog_val.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model
### Testing
    python correct_text.py --test_path /movie_dialog_test.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model \
                       --decode
                    
### Process
在模型训练的时候会生成日志文件，启动tensorboard，PROJECTOR栏将展示投影后的数据的动态图，如下图
        ![图片alt](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task1/5/img/1.png)             
                       
## Improvement
### From the perspective of accuracy
本项目主要使用了LSTM和RNN神经网络，因此，从这个角度来说，可以改进的方法有
-   增加LSTM神经网络神经元的数量，提高神经网络的复杂度，进而提高准确率
-   准确初始化权重，进而提高最后输出的准确率
-   增加神经网络的层数
-   拓展训练集，进而提高准确率

                       
                    





