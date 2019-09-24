Convolutional Neural Network with Word Embeddings for Chinese Word Segmentation
====
简介
---
该项目设计是为了解决已有的中文分词算法中存在的两个主要缺点。通过对已有项目的分析和缺点的认知，从而设计出为了自动捕获n-gram特征和尽可能利用全文信息来进行中文分词的算法。


算法设计简介
---
针对n-gram，该项目设计了一套不需要任何特征工程就能捕捉到丰富的n-gram特征的卷积神经模型。  
即模型conv-seg，模型conv-seg已经自动捕获了丰富的bigram特征。当bigram特征被显式地添加时，模型倾向于过度拟合。克服过度拟合的一个可行方法是引入先验知识。通过直接对大型未标记数据进行预训练的bigram嵌入来引入先验知识。该项目将未标注的文本转换为bigram序列，然后应用word2vec直接对二进制文本嵌入进行预处理。  
而对于对全文信息的利用，该项目提出了另一种有效的方法来将所提出的模型与字嵌入相结合。  
不仅是将预先训练过的词嵌入，该项目还从大型自动划分的数据构建了一个词汇表。两者都对工作产生了积极影响。且两者对于训练的贡献是大致相等的。



**源代码环境部署**
===
原作者推荐在ubuntu环境下使用gpu进行。


>1.在虚拟机ubuntu的环境下安装anaconda，然后安装python2.7以及实验所需要的对应库。(tensorflow>1.0)    
>2.下载[data.zip](https://drive.google.com/open?id=0B-f0oKMQIe6sQVNxeE9JeUJfQ0k)然后解压提取，安装至项目文件夹下。使其结构为：  
>>
	convseg
	|	data
	|	|	datasets
	|	|	|	sighan2005-pku
	|	|	|	|	train.txt
	|	|	|	|	dev.txt
	|	|	|	|	test.txt
	|	|	|	sighan2005-msr
	|	|	|	|	train.txt
	|	|	|	|	dev.txt
	|	|	|	|	test.txt
	|	|	embeddings
	|	|	|	news_tensite.w2v200
	|	|	|	news_tensite.pku.words.w2v50
	|	|	|	news_tensite.msr.words.w2v50
	|	tagger.py
	|	train_cws.py
	|	train_cws.sh
	|	train_cws_wemb.sh
	|	score.perl
	|	README.md
  
>3.根据作者给出的readme.md,我们可以通过使用train_cws.sh文件来进行对主程序的调用来进行训练,例如:
>>$ ./train_cws.sh pku 0 或者 ./train_cws.sh msr 0  

>然后生成训练结果保存于model-pku或model-msr中。  

实验结论
---
在PKU和MSR两种基准数据集上对模型进行了评价。在没有任何特征工程的情况下，该模型在数据集PKU上获得95.7%的竞争得分，在数据集MSR上获得97.3%的竞争得分。

<h2>

参考
====
>>原作者论文：https://arxiv.org/pdf/1711.04411.pdf
