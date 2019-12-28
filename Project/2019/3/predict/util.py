import warnings
import time
warnings.filterwarnings('ignore')



def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))


def log_shape(train, test):
    log('Train data shape: %s' % str(train.shape))
    log('Test data shape: %s' % str(test.shape))






