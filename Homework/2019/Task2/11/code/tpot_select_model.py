from tpot import TPOTClassifier
import train_model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, stdsc = train_model.train_test_init('domain_01.csv')
    tpot=TPOTClassifier(generations=5, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('../model/tpot_pipeline.py')
