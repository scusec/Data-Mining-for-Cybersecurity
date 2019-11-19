### webshell

webshell是web的一个管理工具，可以对web服务器进行操作的权限，也叫webadmin。

webshell一般是被网站管理员用于网站管理、服务器管理等一些用途，但是由于webshell的功能比较强大，可以上传下载文件，查看数据库，甚至可以调用一些服务器上系统的相关命令。黑客通过一些上传方式，将自己编写的webshell上传到web服务器的页面的目录下，然后通过页面访问的形式进行入侵，或者通过插入一句话连接本地的一些相关工具直接对服务器进行入侵操作。
#### 环境
MACOS 10.14.3
python 3.6.8

#### webshell的分类

可以分为以下几个类：

- PHP脚本木马
- ASP脚本木马
- 基于.NET的脚本木马
- JSP脚本木马

### 特征提取

#### 词袋 & TF-IDF模型

把一个PHP文件作为一个完整的字符串处理，定义函数load_one_file加载文件到一个字符串比纳凉中返回。并且开源软件的目录结构相对复杂，所以需要递归访问并且指定目录并加载指定文件

```python
def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('.php') or filename.endswith('.txt'):
                fulepath = os.path.join(path, filename)
                t = load_file(fulepath)
                files_list.append(t)
    return files_list
```

对样本的个数进行统计，并且将webshell样本标记为1，利用2-gram提取词袋模型，并且使用tf-idf进行处理

```python
CV = CountVectorizer(ngram_range=(2, 2),
decode_error="ignore",max_features=max_features,
token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
x=CV.fit_transform(x).toarray()
transformer = TfidfTransformer(smooth_idf=False)
x_tfidf = transformer.fit_transform(x)
x = x_tfidf.toarray()
```

#### opcode&n-gram特征

首先需要使用VLD处理php文件，把处理的结果保存在字符串中，如果使用python2，则使用commands进行命令调用，如果使用python3+，则需要使用替代库subprocess，PHP的opcode都是由大写字母和下划线组成的单词，使用findall函数从字符串中提取全部满足条件的opcode，并以空格连接成一个新字符串

```python
tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)
t = " ".join(tokens)
```

遍历读取指定目录下全部PHP文件，保存其对应的opcode字符串，然后再对opcode字符串进行2-gram处理和tf-idf处理

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/%E6%8F%90%E5%8F%96opcode.png)

### 系统框架图
![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/frame-1.png)

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/frame-2.png)
### 模型训练

本次实验选择了XGBoost、RandomForest和SVM进行测试，其中xgboost和随机森林表现出了比较好的效果，其中运行的结果截图如下：

xgboost

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/xgboost.png)

randomforest

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/wordbag&2-gram_rf.png)

mlp

![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/mlp.png)

CNN
![](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task7/5/screen/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-11-19%20%E4%B8%8B%E5%8D%888.47.53.png)

### 结果

最终结果表明基于以上几种方式的准确率都比较高



### 测试

测试时指定目录，然后按照webshell.py文件中特征处理的方式进行处理，然后调用模型预测即可。



