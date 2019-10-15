import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import resample
# import all the model that we need
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
# import pca and lda
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# to report train result
from sklearn.metrics import classification_report
# to save model
from sklearn.externals import joblib


def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    model = joblib.load(filename)
    return model

def train_test_init(filename):
    df = pd.read_csv(filename, engine='python', header=None)
    df.dropna(axis=0, how='any')

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    X_upsampled, y_upsampled = resample(X[y == 0],
                                        y[y == 0],
                                        replace=True,
                                        n_samples=X[y == 1].shape[0],
                                        random_state=123)
    X = np.vstack((X[y==1], X_upsampled))
    y = np.hstack((y[y==1], y_upsampled))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    stdsc = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test = stdsc.transform(X_test)

    save_model(stdsc, '../model/stdsc.pkl')

    return X_train, X_test, y_train, y_test, stdsc

def init_model():    
    # decision_tree
    decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)

    # bagging
    bagging = BaggingClassifier(base_estimator=decision_tree, n_estimators=100,
                            max_samples=1.0, max_features=1.0,
                            bootstrap=True, bootstrap_features=False,
                            random_state=1)   

    # random_forest
    random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto')

    # extra_tree
    extra_tree = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)

    # adaboost
    adaboost = AdaBoostClassifier(base_estimator=decision_tree,
                                n_estimators=100,
                                learning_rate=0.1,
                                random_state=1)

    # gradient_tree_boosting
    gradient_tree_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

    # voting_classifier
    voting_classifier = VotingClassifier(estimators=[('random_forest', random_forest),
                                                    ('extra_tree', extra_tree), 
                                                    ('gradient_tree_boosting', gradient_tree_boosting)], voting='hard')
    
    return decision_tree, bagging, random_forest, extra_tree, adaboost, gradient_tree_boosting, voting_classifier

class Model:
    def __init__(self, model, stdsc, X_train, y_train, X_test, y_test):
        self.model = model
        self.stdsc = stdsc
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def lda(self):
        lda = LDA()
        self.X_train = lda.fit_transform(self.X_train)
        self.X_test = lda.transform(self.X_test)

    def pca(self):
        pca = PCA()
        self.x_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
    
    def fit(self):
        self.model = self.model.fit(self.X_train, self.y_train)
    
    def result(self):
        print('\nClassification_report:\n')
        print('train:')
        print('    accuracy: %.2f' % self.model.score(self.X_train, self.y_train))
        print('\ntest:')
        classes = ['good', 'bad']
        y_test_pred = self.model.predict(self.X_test)
        result_=classification_report(self.y_test, y_test_pred, target_names = classes)
        print(result_)
        print('    accuracy: %.2f' % self.model.score(self.X_test, self.y_test))

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, stdsc = train_test_init('../data/dataset.csv')
    decision_tree, bagging, random_forest, extra_tree, adaboost, gradient_tree_boosting, voting_classifier = init_model()
    print('\nPlease choose a model to train:')
    print('1.bagging  2.random_forest  3.extra_tree  4.adaboost\n'
            '5.gradient_tree_boosting  6.voting_classifier  0.quit')
    model_dict = {'1':bagging, '2':random_forest, '3':extra_tree,
                '4':adaboost, '5':gradient_tree_boosting, '6':voting_classifier}
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
        
    model = model_dict[choice]
    classifier = Model(model, stdsc, X_train, y_train, X_test, y_test)
    print('\nTraining...')
    classifier.fit()
    classifier.result()
    save_model(classifier.model, '../model/'+filename_dict[choice])
