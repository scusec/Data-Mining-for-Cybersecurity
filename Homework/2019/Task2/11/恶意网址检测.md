# 恶意网址检测
恶意网站是网络犯罪活动的基石。目前，针对恶意的URL的鉴别最有效的方法是黑名单法，即手动构建列表，并在其中列出所有恶意网页的URL。但考虑到现今网络上的恶意网站数量之大，使用此方法在列表中搜索会显得过于繁琐，且由于web链接的不断增长，列表也无法进行实时更新。因此，本项目中，我们运用机器学习技术对恶意网站进行鉴别，能够满足以下需求：
- 打印出模型的准确率和召回率
- 可以根据输入的URL自动判别其安全性
## 数据集

http://s3.amazonaws.com/alexa static/top 1m.csv.zip

http://www.malwaredomainlist.com

## 特征选择
本项目基于以下特征对URL的安全性进行判断：
- URL长度
- 是否包含'.'
- 是否包含'//'
- 是否包含'@'
- 是否包含'&'
- 是否包含子目录
- 是否包含子域名
- 域名长度
- 是否是IP地址
- 是否包含可疑的TLD
- 是否包含可疑的域名

## 算法选择
参考[https://github.com/surajr/URL-Classification/blob/master/URL%20Classification.ipynb](https://github.com/surajr/URL-Classification/blob/master/URL Classification.ipynb)发现使用决策树的效果比较好，所以使用了决策树训练模型

由于使用的测试数据中，正常网址和恶意网址的区别较大，所以模型的准确率较高，准确率为92.560640%，召回率为91.372843 %。

## 测试结果
测试：www.baidu.com - Benign URL

​			http://dinas.tomsk.ru/err/?paypal.ch/ch/cgi-bin/webscr1.htm?cmd=_login-run&dispatch=5885d80a13c0db1f1ff80d546411d7f8a8350c132bc41e0934cfc023d4r4ere32132 - Malicious URL