# README

## 文件结构
C:.             
│  main.ipynb   
│               
└─data          
    ├─black-list
    └─white-list

## 数据集

### webshell
https://github.com/tennc/webshell
https://github.com/ysrc/webshell-sample
https://github.com/xl7dev/WebShell

### PHP-normal
https://github.com/WordPress/WordPress
https://github.com/typecho/typecho
https://github.com/phpmyadmin/phpmyadmin
https://github.com/laravel/laravel
https://github.com/top-think/framework
https://github.com/symfony/symfony
https://github.com/bcit-ci/CodeIgniter
https://github.com/yiisoft/yii2

### 数据集预处理
这里由于webshell好多名字与内容一致，所以采用计算MD5值的方式计算文件哈希进行去重，并且对文件名进行MD5，防止重名
正常PHP文件这里Windows会自动重命名并替换，可不做考虑

这里采用VLD把PHP代码转化为opcode代码，编译成中间层代码，从而提取出了最主要的底层代码特征

### PHP opcode
```
1. 下载 vld.dll 插件并存放在php ext 目录下
http://pecl.php.net/package/vld/0.14.0/windows
2. 配置 php.ini 激活vld.dll 文件
php.ini中添加如下：
extension=php_vld.dll
```
通过
```
php -dvld.active=1 -dvld.execute=0 php_file
```
可以提取出opcode代码

## 训练

### 模型
```
word embedding + lstm方法来建模
embedding使用word2vec方法
代码实现使用python gensim库
使用word2vec算法生成对应元素的数值矩阵
```

### 数据集划分
```python
x_train = np.concatenate((w2v_word_list[0:2000],w2v_word_list[2500:7000]))
y_train = np.concatenate((label_list[0:2000] , label_list[2500:7000]))
x_test = np.concatenate((w2v_word_list[2000:2500] , w2v_word_list[7000:]))
t_test = np.concatenate((label_list[2000:2500] , label_list[7000:]))
```

### 建模

使用keras搭建神经网络
采用单层LSTM模型
使用sigmod函数作为激活函数
adam作为优化器

### 训练结果
Accuracy:0.96 Loss:0.1036

### 测试结果
Accuracy:0.992688 Loss:0.0669