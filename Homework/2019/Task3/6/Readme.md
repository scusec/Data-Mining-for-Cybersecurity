## 基于机器学习的XSS分析检测

### 项目环境：

---

1. Python 3.5
2. **项目训练所使用的数据集**: https://github.com/das-lab/deep-xss
3. 编辑器：**jupyter notebook**

### 项目所使用的特征

---

**这些特征是根据常见的XSS 攻击的payload提取出来的：**

- ‘ script ’的数目
- ‘ java ’的数目
- ‘ iframe ’的数目
- ‘ style ’的数目
- ‘ alter ’的数目
- ‘ （ ’的数目
- ‘ ） ’的数目
- ‘  <  ’的数目
- ‘ > ’的数目
- ‘ % ’的数目
- ‘ \“ ’的数目
- ‘  \\' ’的数目

### 实验步骤：

---

1. 对初始数据进行清洗：由于本次的实验集已经将两种不同的payload分成两个文件，因此我们可以自己在将两个数据集合并时同时添加标签，并使用我们想要提取的特征进行构建特征集。
2. 选择算法：本次使用的是随机森林（Random Forest）和决策树（Decision tree）
3. 构建相应的混淆矩阵，通过混淆矩阵计算准确率和召回率。
4. 打印出相应的准确率和召回率。
5. 使用单个payload对模型进行测试。

### 结果分析

---

- 在Decision tree算法中，accuracy = 0.98, recall = 1.00

  ![决策树](https://github.com/scusec/Data-Mining-for-Cybersecurity/blob/master/Homework/2019/Task3/6/Screen/%E5%86%B3%E7%AD%96%E6%A0%91.png)

- 在Random Forest算法中，accuracy = 0.98, recall = 1.00

  ![随机森林](C:\Users\lenovo1\Desktop\Data-Mining-for-Cybersecurity\Task3\6\Screen\随机森林.png)

