SQLinjection  detection
=============  

1.项目简介  
----  
该项目主要通过对收集的正常输入内容和具有sql注入嫌疑的内容进行采集学习来生成一个能够对输入内容进行自动判断是否具有sql注入嫌疑的分类器。    

2.实验环境以及采用的主要库
----  
anconda 3.0  
spyder  
sklearn  

3.实验内容
----  
实验数据集:https://github.com/flywangfang258/ML-for-SQL-Injection  
通过对选择的特征进行采集，选择特征包括：  
1.传入参数的长度  
2.关键词的出现次数(['and','or','xor','sysobjects','version','substr','len','substring','exists','mid','asc','inner join','xp_cmdshell','version','exec','having','union','order','information schema','load_file','load data infile','into outfile','into dumpfile']
)  
3.信息熵  
4.大写字母的出现频率  
5.特殊字符的出现次数  
6.数字的出现频率  

然后进过决策树训练后，进行交叉验证，输出准确率得分。  
在最后输出测试训练集的准确率、精确率和召回率。并且将训练模型保存。  
但是实验结果表现出了过拟合，对于短的纯字符输入判断会出现错误。  

4.实验结果
----  
交叉验证结果：  
[0.95588972 0.98696742 0.99348371 0.99197995 0.99398195]  
Training score:1.000000  
Testing score:0.991647  
Testing precision score:0.991647  
Testing precision score:0.989312  
Testing recall score:0.993960  

5.参考链接
----
https://github.com/flywangfang258/ML-for-SQL-Injection  
