# 有关MDL_Data_Exploration.ipynb的修改记录

- 阅读以下内容前，请先移步至[移植指南](https://github.com/ChanthMiao/data_hacking/blob/master/README.md)
- 按照以下方法修改，可适配python3.6。

## ./MDL_Data_Exploration.ipynb文件

- 自行替换所有print语句至3.6版本；
- cell[1]，添加以下语句：

  ```python
  from matplotlib import pylab
  from matplotlib import pyplot as plt
  import sys
  import os
  sys.path.insert(0, '..')
  ```

- 在cell[4]为cvs读取方法指定编码：

  ```python
  dataframe = pd.read_csv(data_url, names=['date','domain','ip','reverse','description',
                        'registrant','asn','inactive','country'], header=None, error_bad_lines=False, low_memory=False, encoding='ISO-8859-1')
  ```

- 在cell[13]添加数据类型过滤条件`int`：
  
  ```python
  dataframe = dataframe.applymap(lambda x: x.strip().lower() if not isinstance(x,(np.float64, int)) else x)
  ```

- 修改cell[14]中的`import`语句：

  ```python
  from urllib.parse import urlparse
  ```

- 修改cell[18]中的`import`语句：

  ```python
  from data_hacking.simple_stats import simple_stats as ss
  ```

- 修改cell[23]中第3条语句的命名参数名称：

  ```python
  pivot = pd.pivot_table(subset, values='count', index=['date'], columns=['description'], fill_value=0)
  ```

- 修改cell[24]中第5条至第第9条语句如下：

    ```python
    plt.figure()
    total_agg.plot(label='New Domains in MDL Database')
    pylab.ylabel('Total Exploits')
    pylab.xlabel('Date Submitted')
    plt.legend()
    ```

- 替换cell[26]中的sort方法为sort_index,替换as_matrix方法为values属性：

  ```python
  corr_df.sort_index(axis=0, inplace=True)
  corr_df.sort_index(axis=1, inplace=True)
  corr_matrix = corr_df.values
  ```

## data_hacking包

### data_hacking包下所有__init__.py文件

- 删除所有写在当中的import语句

### data_hacking/simple_stats/simple_stats.py文件

- 自行替换所有print语句至3.6版本；
- 第112行，替换order方法为sort_values方法：

  ```python
  cols_to_keep += dataframe.iloc[r].sort_values(ascending=False).head(matches).index.tolist()[1:]

  ```

## 提供vscode可用的特殊py文件

**移步至[项目仓库](https://github.com/ChanthMiao/data_hacking/blob/master/mdl_exploration/MDL_Data_Exploration.py)**

- 内容和jupyter用文件的不同仅有两点：
  - 由于vscode可以将解释器的执行路径固定在项目根目录，故无效在源码手动修改环境变量path。
  - 同样由于vscode可以将解释器的执行路径固定在项目根目录，相关文件读写路径稍作修改，以项目根目录作为相对路径的参照。
