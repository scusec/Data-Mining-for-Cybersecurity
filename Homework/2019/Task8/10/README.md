# 验证码识别



### 实验环境

- Ubuntu    18.04
- Keras    2.3.1
- Tensorflow    1.13.1



### 文件目录

```
  ├── README.md                         
  ├── code                             // 存放程序代码文件
  │   ├── get_dataset.py             
  │   ├── predict.py                
  │   └── train.py                    
  ├── dataset                          // 存放生成的训练集
  ├── output                           // 存放训练过程的生成框图
  ├── ROOT                             // 存放服务器的配置文件
  ├── processing                       // 存放训练模型过程中的信息
  │   ├── 1.txt                        // 两次训练train.py的输出   
  │   └── 2.txt                        // 第二次训练predict.py的输出
  └── screen                           // 存放程序的运行截图                   └── precict.png    
```



### 实验说明

- 实验环境需要在Ubuntu上运行
- 实验中虽然训练的准确率只达到0.3左右，但是识别的成功率达到0.9左右