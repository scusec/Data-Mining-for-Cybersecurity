import train_model
import create_dataset as feature
import numpy as np
import os
import sys

def predict(domain, model):
    feature_list = feature.calFeatures(domain)
    feature_np = np.array([feature_list])
    feature_np = feature_np.astype(np.float64)
    stdsc = train_model.load_model('../model/stdsc.pkl')
    feature_std = stdsc.transform(feature_np)
    label = model.predict(feature_std)
    print()
    if label==[0]:
        print('result: good')
    elif label==[1]:
        print('result: bad')
    else:
        print('classify error')
    return label

if __name__ == '__main__':
    print('\nPlease choose a model to predict domain:')
    print('1.bagging  2.random_forest  3.extra_tree  4.adaboost\n'
            '5.gradient_tree_boosting  6.voting_classifier  0.quit')
    filename_dict = {'1':'bagging.pkl', '2':'random_forest.pkl', '3':'extra_tree.pkl',
            '4':'adaboost.pkl', '5':'gradient_tree_boosting.pkl', '6':'voting_classifier.pkl'}
    
    while True:
        choice = input()
        if choice in ['1', '2', '3', '4', '5', '6']:
            break
        elif choice=='0':
            print('quit')
            sys.exit()
        else:
            print('input error')

    file = '../model/'+filename_dict[choice]
    flag = os.path.exists(file)
    if flag==True:
        model = train_model.load_model(file)
        domain = input('\nPlease input the domain that you want to detection:\n')
        predict(domain, model)
    else:
        print('\nPlease train this model before using or use another model to predict')
