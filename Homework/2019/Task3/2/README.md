# Assignment 3: XSS Dection

### 使用支持向量机模型进行XSS攻击检测	第2组 

- #### 依赖库：

  ```python
  pandas, sklearn
  ```

- #### 特征选择：
  
  1. payload中script的计数
  2. payload中java的计数
  3. paylaod中"<"的计数
  4. payload中“(”的计数
  5. payload中alert的计数
  6. payload中%的计数（主要是为了检测编码）
  7. payload中是否含有大写字母
  8. payload中是否含有敏感词（比如f**k）
  
- #### 数据集选择

  ```
  https://github.com/das-lab/deep-xss
  ```

  其中dmzo_normal.csv为正常payload样本，xssed.csv为XSS的payload样本，二者均只有payload一列，将数据使用pandas倒入之后进行手动打标签即可；

- #### 使用模型

  SVM支持向量机，kernel选用线性核

- #### 准确率

  98%~99%
