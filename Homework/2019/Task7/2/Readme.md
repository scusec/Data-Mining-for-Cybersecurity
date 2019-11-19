# 基于决策树模型的静态Webshell 检测

#### Task 7  Group 2

- 数据集：

  - 本次数据集均来源于GitHub公开的数据集，其中正常样本为收集的php项目代码文件
  - 数据集csv链接（已清洗+打标签）：https://pan.baidu.com/s/1pEcxwDYrxspOe2qqixStgA
  - Webshell源码：3286份，正常php源码：5200份；

  

- 特征选取：

  ​	1. 代码行数

  ​	2. 信息熵

  3. 代码中关于string操作的相关函数（如trim、echo等）；
  
   	4. 代码中sys操作的函数（eval、exec、popen等）；
   	5. 代码中最长字符串的长度；
   	6. 非单字母单词数量；
   	7. code总长度；
   	8. code中GET和POST参数个数；

 

- 模型及评估
  - 决策树
    - Precision：0.96
    - Recall：0.95
    - F1-Score：0.96



- 运行
  - 本模型直接集成到了文件model.ipydb下，直接使用IPython运行即可；
  - 数据集因源代码文件过多且过大，这里只放出了清洗之后的csv文件，IPython有一些读取php文件的模块不需要运行（比较好看出来，也有注释）；