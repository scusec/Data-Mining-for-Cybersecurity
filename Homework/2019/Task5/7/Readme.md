# SQL注入

## 什么是SQL注入

​		SQL注入即是指web应用程序对用户输入数据的合法性没有判断或过滤不严，攻击者可以在web应用程序中事先定义好的查询语句的结尾上添加额外的SQL语句，在管理员不知情的情况下实现非法操作，以此来实现欺骗数据库服务器执行非授权的任意查询，从而进一步得到相应的数据信息。
​		SQL是操作数据库数据的结构化查询语言，网页的应用数据和后台数据库中的数据进行交互时会采用SQL。而SQL注入是将Web页面的原URL、表单域或数据包输入的参数，修改拼接成SQL语句，传递给Web服务器，进而传给数据库服务器以执行数据库命令。如Web应用程序的开发人员对用户所输入的数据或cookie等内容不进行过滤或验证(即存在注入点)就直接传输给数据库，就可能导致拼接的SQL被执行，获取对数据库的信息以及提权，发生SQL注入攻击。 



### 原理

​		SQL注入攻击是通过操作输入来修改SQL语句，用以达到执行代码对WEB服务器进行攻击的方法。简单的说就是在表单、输入域名或页面请求的查询字符串中插入SQL命令，最终使web服务器执行恶意命令的过程。可以通过一个例子简单地说明一下SQL注入攻击。假设某网站页面显示时URL为http://www.test.com?test=123，此时URL实际向服务器传递了值为123的变量test，这表明当前页面是对数据库进行动态查询的结果。由此，我们可以在URL中插入恶意的SQL语句并进行执行。另外，在网站开发过程中，开发人员使用动态字符串构造SQL语句，用来创建所需的应用，这种情况下SQL语句在程序的执行过程中被动态的构造使用，可以根据不同的条件产生不同的SQL语句，比如需要根据不同的要求来查询数据库中的字段。这样的开发过程其实为SQL注入攻击留下了很多的可乘之机。 



### 示例 

- -290 union select 1,2,3,4,5,6,concat_ws(0x3a3a,id,uid,datum,count,user,stamp),8,9,10 from a_logins -- 

- 295 union select 1,2,3,4,5,6,7,8,9,10 from msysaccessobjects

- 2 and 1=0 UnIon selECt 1,2,3,version(),5,6,7,8,9,10,11,12,13,14,15,16,17 -- 

  

## 实现流程

### 原始数据预处理

1. 将sql注入的标签设置为1，正常标签设置为0。

2. 对url进行解码。

3. 去重，删除相似类型数据。

4. 去除前面无用部分，只留下第一个?后的payload部分。

   

### 特征提取

1. payload长度

2. payload中数字出现频率

3. payload中大写字母出现频率

4. payload中空格出现频率

5. payload中出现的关键字数量，包括： "select", "from", "insert", "delete", "having", 
       "union", "count", "drop table", "update",
       "truncate", "asc", "mid", "char", "xp_cmdshell", 
       "exec", "master", "net", "and", "or", "where", "substr", 
       "information schema", "xor", "version", "set",
       "where", "group", "order", "create", "sum", "max",
       "min", "avg", "having", "except"
   
   

### 归一化处理

1. 对提取好的特征使用最大最小归一化算法进行归一化处理，以提升在Logistics回归模型中的收敛速度和模型精度。

2. 但是在使用决策树模型时，最好使用没有归一化处理过的数据，所以我们同时也准备了没有归一化处理过的数据。

   

### 基于不同算法进行训练

| 算法         | 训练准确度 | 测试准确度 | 召回率 | 交叉验证 |
| ------------ | ---------- | ---------- | ------ | -------- |
| 决策树       | 99.91%     | 99.625%    | 99.6%  | 99%      |
| 逻辑斯蒂回归 | 96.64%     | 96.625%    | 95.9%  | 97%      |

输入参数检验：

1. "https://github.com/scusec/Data-Mining-for-Cybersecurity/tree/master/Homework/2019/Task5"；
2. "http://localhost/sqlilabs/Less-2/?id=-1 union select 1,2,SCHEMA_NAME, from information_schema.SCHEMATA limit 1,1"；

判断结果：第一条为正常url，第二条为sql注入，结果正确。