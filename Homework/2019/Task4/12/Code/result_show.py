import itertools
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import lstm_train
from scipy import interp
from sklearn.metrics import roc_curve, auc

def create_figs(nfolds=10, force=False):
    if force or (not os.path.isfile('../Model/results.pkl')):
        results = lstm_train.run(nfolds=nfolds)
    else:
        results = pickle.load(open('../Model/results.pkl', 'rb'))

    fpr = []
    tpr = []
    for result in results:
        t_fpr, t_tpr, _ = roc_curve(result['y'], result['probs'])
        fpr.append(t_fpr)
        tpr.append(t_tpr)
    binary_fpr, binary_tpr, binary_auc = calc_macro_roc(fpr, tpr)

    from matplotlib import pyplot as plt
    with plt.style.context('bmh'):
        plt.plot(binary_fpr, binary_tpr,
                 label='LSTM (AUC = %.4f)' % (binary_auc, ), rasterized=True)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        plt.title('ROC - Binary Classification', fontsize=26)
        plt.legend(loc="lower right", fontsize=22)

        plt.tick_params(axis='both', labelsize=22)
        plt.savefig('../Screen/results.png')

def calc_macro_roc(fpr, tpr):
    all_fpr = sorted(itertools.chain(*fpr))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)

if __name__ == "__main__":
    create_figs(nfolds=1) 
