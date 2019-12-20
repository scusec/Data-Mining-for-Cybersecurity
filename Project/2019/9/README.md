# DGA Prediction Using LSTM Based On ATTENTION
### 一、数据集

白样例：

 Alexa全球域名点击排名

 https://amazonaws-china.com/cn/alexa-top-sites/

黑样例：

 http://data.netlab.360.com/feeds/dga

 http://osint.bambenekconsulting.com/feeds

### 二、 数据格式

    (一) 二分类
    
        (1) 训练数据
            1. 一行一组数据
            2. 格式：域名 类型
            3. 0代表非DGA域名，1代表DGA域名
    
            eg:
                y71f2169b8572150ffd4ac497a5f38c801.hk 1
                timoneuwtihe.ddns.net 1
                l2c89945113a3984626c14d00778abb7d7.cn 1
                taojindi.com 0
                ggmee.com 0


        (2) 测试数据
            1. 同训练数据，不过没有类型
    
            eg:
                cobjjozfxzadnvzvn.net
                ezqpqtude.com
                jveugx.com
                edujgzyd.org
                abhhlyftdpxa.dyn


        (3) 返回结果
            1. 一行一个结果，对应测试数据，结果是float类型，对应着相应域名是由DGA算法生成的概率
    
            eg:
                0.6172037720680237
                0.9998446702957153
                0.9998708963394165
                0.5719057321548462




    (二) 多分类
    
        (1) 训练数据
            1. 一行一组数据
            2. 格式：域名 类型
    
            eg:
                194ozgcrfyuyi1dl15gqeukjc6.org 16
                rvitghrkweqrnur.biz 13
                bqoxixuyr.mn 14
                tfwafordlinnetavox.com 8
                tumboor.com 0



        (2) 测试数据
            1. 同训练数据，不过没有类型
    
            eg:
                emsesgumwwbvtcbh.eu
                transrush.com
                bbrcqtndpjlt.com
                monyer.com
                vatoefurex.ddns.net


        (3) 返回结果
            1. 一行一个结果，对应测试数据，结果是相应域名的类型
    
            eg:
                77
                24
                0
                3
                0

### 三、 程序使用

1. DGA_prediction_LSTM+ATTENTION为模型训练代码

2. web为前端代码（static里面图片太多，无法直接上传，故进行了压缩）

    (一) 二分类训练
        命令行：python main.py 0 batch_size epochs dataFilePath modelFilePath
    
        参数说明：
            batch_size 批处理大小
            epochs  训练轮数
            dataFilePath : 训练数据文件路径
            modelFilePath : 训练好的模型文件保存路径，是个文件路径
        
        举例：
            python main.py 0 100 1 /home/audr/chc/data/Binary/11.22/train_11.22.txt /home/audr/chc/models/model_100_1_b.h5
        说明：
            1. 进行二分类训练，批处理大小为100，数据训练轮数为1
            2. 训练数据文件路径：/home/audr/chc/data/Binary/11.22/train_11.22.txt
            3. 训练模型保存路径： /home/audr/chc/models/model_100_1_b.h5



    (二) 二分类测试
        命令行：python main.py 1 batch_size epochs dataFilePath modelFilePath resultFilePath
    
        参数说明：
            batch_size 批处理大小
            epochs  训练轮数，测试时这个参数其实没有用到，只是为了统一参数的格式
            dataFilePath : 测试数据文件路径
            modelFilePath : 加载模型路径
            resultFilePath : 判别结果保存路径，是个文件路径
    
        举例：
            python main.py 1 100 1 /home/audr/chc/data/Binary/11.22/test_11.22.txt /home/audr/chc/models/model_100_1_b.h5 /home/audr/chc/result/test_result_100_1_b.txt
        说明：
            1. 进行二分类测试，批处理大小为100
            2. 测试数据文件路径：/home/audr/chc/data/Binary/11.22/test_11.22.txt
            3. 测试模型路径： /home/audr/chc/models/model_100_1_b.h5
            4. 判别结果文件路径：/home/audr/chc/result/test_result_100_1_b.txt


    (三) 多分类训练
        命令行：python main.py 2 batch_size epochs dataFilePath modelFilePath nb_classes
    
        参数说明：
            batch_size 批处理大小
            epochs  训练轮数
            dataFilePath : 训练数据文件路径
            modelFilePath : 训练好的模型文件保存路径，是个文件路径
            nb_classes : 训练数据中数据的类型个数，比如：正常域名 + 5种DGA feed生成的域名  nb_classes = 6
    
        举例：
            python main.py 2 100 1 /home/audr/chc/data/Multiclass/11.22/train_11.22_10000.txt /home/audr/chc/models/model_100_1_m.h5 20
        说明：
            1. 进行多分类训练，批处理大小为100，数据训练轮数为1
            2. 训练数据文件路径：/home/audr/chc/data/Multiclass/11.22/train_11.22_10000.txt
            3. 训练模型保存路径： /home/audr/chc/models/model_100_1_m.h5
            4. 域名类别总个数nb_classes：20
    
    (四) 多分类测试
        命令行：python main.py 3 batch_size epochs dataFilePath modelFilePath resultFilePath
    
        参数说明：
            batch_size 批处理大小
            epochs  训练轮数，测试时这个参数其实没有用到，只是为了统一参数的格式
            dataFilePath : 测试数据文件路径
            modelFilePath : 加载模型路径
            resultFilePath : 判别结果保存路径，是个文件路径
    
        举例：
            python main.py 3 100 1 /home/audr/chc/data/Multiclass/11.22/test_11.22_10000.txt /home/audr/chc/models/model_100_1_m.h5 /home/audr/chc/result/12.13/test_result_100_1_m.txt
        说明：
            1. 进行多分类测试，批处理大小为100
            2. 测试数据文件路径：/home/audr/chc/data/Multiclass/11.22/test_11.22_10000.txt
            3. 测试模型路径： /home/audr/chc/models/model_100_1_m.h5
            4. 判别结果文件路径：/home/audr/chc/result/12.13/test_result_100_1_m.txt

```
(五) web前端
	命令行：python app.py
```

