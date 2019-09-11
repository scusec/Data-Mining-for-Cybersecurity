# mdl_exploration

这篇文档讲述了如何部署 mdl_exporation 的环境，并且成功运行。附上了 requirements 文件和修改过的 notebook 文件。

**写在前面，使用 python2！！！**

## Requirements

首先需要安装相关依赖，相关依赖在 setup.py 文件的第 13 行已经写明了

```python
install_requires=[ 'networkx','pygraphviz','pandas','matplotlib','numpy','tldextract','sqlparse','macholib','pefile','patsy','statsmodels','sklearn' ],
```

这里给大家提供了一个 requirements 文件，运行`python -m pip install -r requirements.txt`即可安装相关依赖。此处的 requirements 文件里没有安装 pygraphviz ，因为在安装的时候遇到了奇怪的问题，可以直接下载 wheel 文件进行安装，**Windows 用户**点击[此处](https://download.lfd.uci.edu/pythonlibs/g5apjq5m/pygraphviz-1.3.1-cp27-none-win32.whl)下载，然后执行`python -m pip install xxxxx.whl`即可，xxxxx 是文件名。

## Install data_hacking

在安装前需要修改一个文件，由于 pandas 版本较高，我的版本为 0.24.2，一些在 0.12 版本的 api 已经被弃用。

- 修改/data_hacking/simple_stats/simple_stats.py 的第 113 行

```python
# 修改前
cols_to_keep += dataframe.iloc[r].order(ascending=False).head(matches).index.tolist()[1:]
# 修改后
cols_to_keep += dataframe.iloc[r].sort_values(ascending=False).head(matches).index.tolist()[1:]
```

- 修改完之后直接这样就能安装了（如果修改的时候发现同名 pyc 文件记得把 pyc 文件先删除）

```shell
python setup.py install
```

- 重启 notebook 的 kernel

## Changes in Notebook

以下 In 的编号以跑一次就通过的编号为准，如果和实际情况不符，请自行定位代码。

1. 在 In[2]中，添加`import pylab`
2. 在 In[4]中，在末尾添加`encoding='ISO-8859-1'`或者将文件编码从 ANSI 转变为 UTF-8，修改后代码如下：

```python
dataframe = pd.read_csv(data_url, names=['date','domain','ip','reverse','description', 'registrant','asn','inactive','country'], header=None, error_bad_lines=False, low_memory=False, encoding='ISO-8859-1')
```

3. 在 In[8]中，添加`import numpy as np`
4. 在 In[13]中，修改代码为：

```python
# 是数值就直接保留，不是数值的话进行处理
dataframe = dataframe.applymap(lambda x: x.strip().lower() if not (isinstance(x,np.float64) or isinstance(x,int) or isinstance(x,float) or isinstance(x,long)) else x)
```

5. 在 In[26]中，添加`from matplotlib import pyplot as plt`，并且修改两处方法，原有方法在新版本中已被弃用，修改后结果如下：

```python
# Statsmodels has a correlation plot, we expect the diagonal to have perfect
# correlation (1.0) but anything high score off the diagonal means that
# the volume of different exploits are temporally correlated.
import statsmodels.api as sm
from matplotlib import pyplot as plt
corr_df.sort_index(axis=0, inplace=True) # Just sorting so exploits names are easy to find
corr_df.sort_index(axis=1, inplace=True)
corr_matrix = corr_df.as_matrix()
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
sm.graphics.plot_corr(corr_matrix, xnames=corr_df.index.tolist())
plt.show()
```
