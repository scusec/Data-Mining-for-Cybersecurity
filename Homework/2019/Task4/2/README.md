# DGA域名检测

1. 数据集：
   - DGA Domain：https://data.netlab.360.com/feeds/dga/dga.txt
   - Normal Domain：https://www.kaggle.com/cheedcheed/top1m

2. 分类器：
   - Keras二分类器
   - 决策树Decision Tree
3. 分类结果：
   - Keras分类方法达到了在测试集合上精确率82%，召回率70%；
   - ![image-20191030124314129](/Users/devin/Library/Application Support/typora-user-images/image-20191030124314129.png)
   - 决策树分类方法达到了在测试集合上精确率92%，召回率89%
   - ![image-20191030123618720](/Users/devin/Library/Application Support/typora-user-images/image-20191030123618720.png)
4. 特征情况：
   - domain长度
   - 元音特征——元音字母占全部字母的比例
   - 辅音特征——连续的辅音串数量占全部字母的比例
   - 数字特征——数字占全部字符比例
   - Domain部分信息熵
   - 十六进制哈希特征——Domain中0-9、a-f总长度的比例
   - 唯一出现过的字母占所有出现过的字母的比例
   - TLD顶级域检测——是否是com或cn
   - n-gram转化的Vector向量
5. 模型评估：
   - 选用keras进行二分类的效果不是很好，且运行速度较慢；在后续修改中采取了决策树的形式，运行快且精确度相对较高；
   - 猜测原因其一是因数据量大（160万条）keras进行模型训练时耗费资源过大，其二是对于DGA域名这种检测方式，信息熵这一因素占据很高的权重，而决策树这一模型的提出正是基于信息熵的，所以在决策树模型上的表现也会高于keras分类器。