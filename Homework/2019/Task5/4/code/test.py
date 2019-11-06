import re
from utils import *
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# x=np.load('bins/x_train.npy')
# print(x.shape)

# y=np.load('bins/y_train.npy')
# print(y.shape)

# with open('bins/classes_voc.pkl','rb') as f:
#     classes=pickle.load(f)
# print(classes)
# print(len(classes))

# print(segment('# #',True))

a=[0,1,2,3,4,5,6]
b=[0,0,0,0,0,0,0]

a_tr,a_te,b_tr,b_te=train_test_split(a,b)
print(a_tr)



