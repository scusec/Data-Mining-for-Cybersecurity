import pickle
from zipfile import ZipFile
import os
import random
import tldextract
import pandas as pd



DATA_FILE = 'traindata.pkl'

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()



def get_data():

	if (not os.path.isfile(DATA_FILE)):
		domains = []
		labels = []
		df = pd.read_csv("alexa-top-1m.csv",header=None)
		df = df[1:]
		df.columns = ['Rank','Domain']
		domains=df.Domain.tolist()
		labels += ['benign']*len(df.Domain)
		df = pd.read_csv("dgadataset.csv",header=None)
		df = df[1:]
		df.columns = ['Domain','Botnet']
		domains+=df.Domain.tolist()
		labels += ['dga']*len(df.Domain)
		print ('Dumping file')
		pickle.dump(zip(labels, domains), open(DATA_FILE, 'wb'))
		print ('Dumping Completed')
	return pickle.load(open(DATA_FILE,'rb'))

get_data()
