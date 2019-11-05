import pandas as pd
from sqli_detect import getFeatures
import time
import pickle

if __name__ == '__main__':
    while 1:
        payload = input("please input your url:")
        print('start process data')
        start = time.time()
        result = pd.DataFrame(columns=('sql','length','key_num','capital_f','num_f','space_f','special_f','prefix_f','entropy','label'))
        results = getFeatures(payload, '1')
        result.loc[0] = results
        result = result.drop(['sql','label'],axis=1).values
        print(result)
        end = time.time()
        print('Over process in %f s'%(end -start))
        with open('models.model','rb') as fr:
            clf = pickle.load(fr)
        print('start Predict job')
        start = time.time()
        print(clf.predict(result))
        end = time.time()
        print('Over Predict job in %f s'%(end - start))