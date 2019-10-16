## 运行环境

- python 3.7
- jupyter notebook

## 依赖

- sklearn
- pandas

## 数据集

https://github.com/das-lab/deep-xss   【dmzo_nomal.csv, xssed.csv】

## 数据处理

### 数据清洗

1. 字符串转化为小写
2. 去掉所有的换行符"\<br/>"

### 特征提取

#### label

将含XSS数据标记为1， 不含XSS数据标记为0

#### feature(启发式)

1. 是否有未闭合的尖括号（is_angle_brackets_closed）
2. 是否含有alert弹窗（has_pop_up_window）
3. 字符串长度（length）
4. 是否嵌入JavaScript（is_script_embedded）
5. 是否有iframe子框（has_iframe）
6. “%”个数（per_cent_sign_num）
7. 是否有反斜杠（has_backslash）
8. 是否有闭合标志' （has_closed_sign）
9. 是否有document.cookie（has_document_cookie）

### 训练方法

- 随机森林 
- 30%数据用作测试集、其余为训练集

### 模型评估

准确率：99.33%

召回率：95.85%

### 使用方法

在弹出框内输入待检测的域名即可得到判别结果

预置测试数据：<img src=\'#\' onerror=javascript:alert(1)>