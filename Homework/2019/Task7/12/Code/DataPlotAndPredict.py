
# coding: utf-8

# In[1]:


import pandas as pd
shellLocation="../data/dealdata.txt"
eduLocation="../data/edudealdata.txt"

edudf = pd.read_csv(eduLocation,sep="\t", header=None,names = ["fsize","ftagnum","flinecount","ftagtype","fflag"])
shelldf=pd.read_csv(shellLocation,sep="\t", header=None,names = ["fsize","ftagnum","flinecount","ftagtype","fflag"])


# In[2]:


edudf.head()


# In[3]:


eduX = edudf[["fsize","ftagnum","flinecount","ftagtype",]]
eduX.tail()


# In[4]:


shellX = shelldf[["fsize","ftagnum","flinecount","ftagtype",]]
shellX.tail()


# In[5]:


eduy = edudf[['fflag']]
shelly = shelldf[['fflag']]


# In[6]:


import pylab as pl

pl.figure(figsize=(8,6),dpi=100)
pl.plot(edudf['ftagnum'],edudf['ftagtype'],'.',color='blue', label='edu')
pl.plot(shelldf['ftagnum'],shelldf['ftagtype'],'.',color='red', label='shell')
pl.xlabel('ftagnum')
pl.ylabel('ftagtype')

pl.legend()
pl.title('total num of tags and total type of tags')
pl.show()


# In[7]:


pl.figure(figsize=(16,12))

pl.plot(edudf['fsize'],edudf['flinecount'],'.',color='blue', label='edu')
pl.plot(shelldf['fsize'],shelldf['flinecount'],'.',color='red', label='shell')
pl.xlabel('fsize')
pl.ylabel('flinecount')

pl.legend()
pl.title('fsize and flinecount')
pl.show()


# In[8]:


# 数据集合划分
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
totalLocation="../data/totaldata.txt"
totoaldf = pd.read_csv(totalLocation,sep="\t", header=None,names = ["fsize","ftagnum","flinecount","ftagtype","fflag"])
totalX = totoaldf[["fsize","ftagnum","flinecount","ftagtype",]]
totaly = totoaldf[['fflag']]
for randomNum in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(totalX, totaly, test_size=0.33, random_state=randomNum)
    clf = GaussianNB()
    #拟合数据
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)
    # calculate accuracy
    print('数据集',randomNum,"::", end="\n")
    print('准确率：',metrics.accuracy_score(y_test, y_pred_class),"\t",end="\n")
    print('精确率：',metrics.precision_score(y_test,y_pred_class),"\t",end="\n")
    print('召回率：',metrics.recall_score(y_test,y_pred_class),"\t",end="\n")
    print('错误率：',1-(metrics.accuracy_score(y_test, y_pred_class)),end="\n")


# In[ ]:





# In[9]:


clf.score(X_test, y_test) 


# In[10]:


y_pred_class = clf.predict(X_test)


# In[11]:


# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
print('准确率：',metrics.accuracy_score(y_test, y_pred_class))
print('精确率：',metrics.precision_score(y_test,y_pred_class))
print('召回率：',metrics.recall_score(y_test,y_pred_class))
print('错误率：',1-(metrics.accuracy_score(y_test, y_pred_class)))


# In[12]:


# 训练感知机模型
from sklearn.linear_model import Perceptron
# n_iter_no_change：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)


# In[13]:


ppn.get_params()


# In[14]:


# 分类测试集，这将返回一个测试结果的数组
y_pred = ppn.predict(X_test)
# 计算模型在测试集上的准确性
metrics.accuracy_score(y_test, y_pred)


# In[15]:


from sklearn import svm
svr = svm.SVR()
svr.fit(X_train, y_train) 


# In[16]:


y_pred = svr.predict(X_test)
svr.score(X_test, y_test, sample_weight=None)

