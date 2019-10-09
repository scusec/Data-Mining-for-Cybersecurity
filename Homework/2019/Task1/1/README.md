# Text Correct 文本纠错
### 简介

- 现在的许多文本编辑器可以很好的检查拼写错误，但是几乎无法检查出哪怕最基本的语法错误，例如：

```
- 错误的句子：I'm going to the school.
- 正确的句子：I'm going to school.

对于我这个习惯用the凑词数的人来说简直太真实了
```
- 这个project的文本纠错正是用来在短消息中修复简单的语法小错误。作者说它可以用来帮助英语学习者，但我觉得它也可以集成到word之类的文本编辑器中。


### 运行环境及使用说明
1. Python版本：3.7.4（Anaconda）
2. 主要依赖的库：
```
 - tensorflow 1.14(包含numpy等依赖包，CPU版本)
 - pandas 0.24.2
 - scikit-learn 0.21.2
```
3. 玄学debug因机器而异，一调就是半天，看缘分

### 模型要素解读
1. 训练集和测试集的选择：
```
root_data_path = "/Users/atpaino/data/textcorrecter/dialog_corpus"
train_path = os.path.join(root_data_path, "movie_lines.txt")
val_path = os.path.join(root_data_path, "cleaned_dialog_val.txt")
test_path = os.path.join(root_data_path, "cleaned_dialog_test.txt")
model_path = os.path.join(root_data_path, "dialog_correcter_model_testnltk")
config = DefaultMovieDialogConfig()
```
- 数据源：[Cornell Movie Dialogs Corpus]（http://www.cs.cornell.edu/~cristian/cornell_movie-dialogs_corpus.html）
- 程序抽取了一定量的没有语法错误的英语文本，并在每个文本中加入一个小的语法错误形成一个错误文本。主要的构造方法为：
```
 - 去掉冠词如（a，an，the）
 - 动词词形变位的增删（如“ve”、“ll”、“s”、“m”）
 - 同音字替换（例如，将“their”替换为“there”，“then”替换为“than”）。
```
- 将错误文本作为输入，正确文本作为输出，并以此训练模型。
- 共243768句用于训练模型，30474句用于测试集。

2. session和train的构造：
```
 - 使用两层，每层512节点的LSTM训练“sequence-to-sequence model”
 - self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
 -              self.encoder_inputs, self.decoder_inputs, targets,
 -              self.target_weights, buckets,
 -              lambda x, y: seq2seq_f(x, y, True),
 -              softmax_loss_function=softmax_loss_function)
```

3. 部分训练结果(其中decoding是构造的错误文本，Target是训练出来的模型对其的修正)：

```
 - Decoding: investment banking . moving money from a place to place .
 - Target:   investment banking . moving money from place to place .

 - Decoding: what 's the c .r .s . ?
 - Target:   what 's c .r .s . ?

 - Decoding: this is a c .r .s .
 - Target:   this is c .r .s .

 - Decoding: their ladder here .
 - Target:   there 's a ladder here .

 - Decoding: this is n't attempt to be gallant . if i do n't lift you , how are you going to get there ?
 - Target:   this is n't an attempt to be gallant . if i do n't lift you , how are you going to get there ?
```

4. 缺点和改进的空间：

 - 目前的模型只能解决人为加入的，简单的语法错误，其改正错误的能力取决于人为它设定了多少种错误。
 - 仅限于短文本，长文本中更为隐秘的上下文联系难以识别。
 - 对于英语中一些使用冠词或助动词来强调或表达特殊含义的句子难以正确识别。
 
 
 
 