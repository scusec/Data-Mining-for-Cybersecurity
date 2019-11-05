# SQL注入检测(SQL-Injection-Detection)

​	此项目使用sklearn中函数进行训练，通过正常域名与SQL域名的数据集，处理选取了9个训练特征，使用了KNN算法的10轮交叉验证，训练准确率为0.9359403862891279；也使用了随机森林、决策树以及逻辑回归等训练算法进行训练，经测试准确率较高。

## 数据集选择

​	此实验选择正常域名数据集以及SQL注入数据集作为训练集，分别含有5000以及4974条数据。以总训练集的随机30%切分作为测试集进行测试。

## 导入数据处理

​	在数据集中导入数据时去除每一条数据的最后一位‘\n'，然后分别加入列表之中。

```python
normal_train = []
sql_train = []
for x in open("./dataset/normal_train.csv",'rb').readlines():
    normal_train.append(unquote(x[:-1].decode()))
for x in open("./dataset/sql_train.csv",'rb').readlines():
    sql_train.append(unquote(x[:-1].decode()))
```

## 特征选择

​	此项目选取了9个特征用于训练模型，分别为长度、数字占比、空格占比、大写字母占比、信息熵、关键字数量、特殊符号占比、前缀占比以及检测引号是否封闭。

​	在计算各个比例是会首先考虑长度是否为0，若是0则跳过此域名：

```python
length = len(domain)
if length == 0:
	continue
```
​	用于判断的关键字，特殊符号以及前缀分别为：

```python
keyword = ['and','or','xor','sysobjects','version','substr','len','substring','exists','mid','asc','inner join','xp_cmdshell','exec','having','union','order','information schema',
'load_file','load data infile','into outfile','into dumpfile','select']
special_char = ['{','}','(',')','NULL','=','?','[',']']
prefix = ['\\x','&','\\u','%']
```

​	信息熵的计算函数为：

```python
def calc_ent(domain):
    dataset = []
    for each in domain:
        dataset.append(each)
    data1 = np.array(dataset)
    x_value_list = set([data1[i] for i in range(data1.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(data1[data1 == x_value].shape[0]) / data1.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent
```

​	判断此域名单引号与双引号是否闭合，闭合则特征值为1，不闭合则为0：

```python
close = (1 if (domain.count("'")%2 == 0) & (domain.count("\"")%2 == 0) else 0)
```

## 标签添加

​	将获取到的特征值与标签合为同一个集合，SQL域名标签值为1，正常域名标签值为0：

```python
features = []
labels = []
for feature in make_feature(sql_train):
    features.append(feature)
    labels.append(1)
for feature in make_feature(normal_train):
    features.append(feature)
    labels.append(0)
```

## 训练模型

#### K阶近邻及其K折交叉验证

​	K近邻训练数据集并使用十轮交叉验证：

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3，random_state=5)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
scores = cross_val_score(knn, features, labels, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())
```

​	训练结果如下，可见十轮交叉验证后准确率的平均值为0.9359403862891279。

```
0.9474739374498797
[0.86172345 0.90881764 0.93086172 0.97294589 0.97893681 0.95787362 0.95085256 0.94583751 0.97592778 0.87562688]
0.9359403862891279
```

#### 随机森林

​	以随机森林算法进行训练：

```python
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.3, random_state=5)
clf = RandomForestClassifier()
clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
print("precision_score:", precision_score(y_true=test_y, y_pred=pred_y))
print("recall_score:", recall_score(y_true=test_y, y_pred=pred_y))
```

​	训练结果如下：

```
precision_score: 0.9993279569892473
recall_score: 0.9906728847435043
```

#### 决策树以及逻辑回归

​	与以上相同均使用sklearn库函数实现，训练结果无差异，故在此不再赘述。

## 测试

​	使用KNN算法训练的模型，通过前端输入进行测试，经测试判断较为准确.

```python
def test_domain_KNN(domain):
    test_feature = []
    test_feature.append(domain)
    sample = make_feature(test_feature)
    pre = knn.predict(sample)
    print("SQL") if pre == 1 else print("Normal")

while True:
    test_domain_KNN(input("KNN测试："))
```

