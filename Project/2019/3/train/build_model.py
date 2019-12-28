from util import *

config_lgb = {
    'rounds': 10000,
    'folds': 5
}

params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'xentropy',
    'metric': 'auc',
    'learning_rate': 0.02,
    'scale_pos_weight': 200, 
    'num_leaves': 31,  
    'max_depth': -1,  
    'min_child_samples': 100,  
    'max_bin': 128,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.9,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0.99,  
    'reg_lambda': 0.9,  
    'nthread': 24,
    'verbose': 1,
    'seed': 8
}



def spilt_vc_train_test(df, train_size, test_size):
    vc_train = df[:train_size]
    vc_test = df[train_size:train_size + test_size]
    return vc_train, vc_test


def get_model_input_data(train, test=None, is_vc=0):
    feat = ['ip', 'app', 'device', 'os', 'channel', 'hour']
    for f in feat:
        if f not in predictors:
            predictors.append(f)
    train_x = train[predictors]
    train_y = train.is_attributed.values
    if is_vc == 1:
        test_x = test[train_x.columns.values]
        test_y = test.is_attributed.values
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y

#categorical_feature 为lightbgm指定哪些是类别特征，不必再转换为onehot编码
def lgb_cv(train_feature, train_label, test_feature, test_label, params, folds, rounds):
    start = time.clock()
    print(train_feature.columns)
    params['scale_pos_weight'] = float(len(train_label[train_label == 0])) / len(train_label[train_label == 1])
    dtrain = lgb.Dataset(train_feature, label=train_label, categorical_feature=['app', 'device', 'os', 'channel', 'hour'])
    dtest = lgb.Dataset(test_feature, label=test_label, categorical_feature=['app', 'device', 'os', 'channel', 'hour'])
    num_round = rounds
    print('LightGBM run cv: ' + 'round: ' + str(rounds))
    res = lgb.train(params, dtrain, num_round, valid_sets=[dtest], valid_names=['test'], verbose_eval=1, early_stopping_rounds=20)
    elapsed = (time.clock() - start)
    print('Time used:', elapsed, 's')
    return res.best_iteration, res.best_score['test']['auc'], res

#categorical_feature 为lightbgm指定哪些是类别特征，不必再转换为onehot编码
def lgb_deploy(train_feature, train_label, rounds, params):
    dtrain = lgb.Dataset(train_feature, label=train_label, categorical_feature=['app', 'device', 'os', 'channel', 'hour'])
    num_round = rounds
    model = lgb.train(params, dtrain, num_round, valid_sets=[dtrain], verbose_eval=1)
    return model


def reload_all_feature():
    global predictors
    global train_len
    log("reloading all.p ......")
    df = cPickle.load(open('all.p', 'rb'))
    log("reload all.p done")

    log("reloading predictors.p ......")
    predictors=cPickle.load(open('predictors.p', 'rb'))
    log("reload predictors.p done")

    log("reloading train_len.p ......")
    train_len=cPickle.load(open('train_len.p', 'rb'))
    log("reload train_len.p done")
    return df

def train_test_supplement_split(df):
    log('Train test_supplement split...')
    train = df[:train_len]
    test_supplement = df[train_len:]
    del df
    gc.collect()
    log_shape(train, test_supplement)
    log('Train test_supplement split done!')
    return train

def split_dataset_for_cv(train):
    log('Split dataset to get vc train/test set...')
    vc_train_size = 1000000  
    vc_test_size = 250000
    vc_train, vc_test = spilt_vc_train_test(train, vc_train_size, vc_test_size)
    log('Split dataset to get vc train/test set done!')
    return vc_train,vc_test

def get_vc_model_input_data(vc_train,vc_test):
    log('Get vc model input data...')
    vc_train_x, vc_train_y, vc_test_x, vc_test_y = get_model_input_data(train=vc_train, test=vc_test, is_vc=1)
    del vc_train
    del vc_test
    gc.collect()
    log_shape(vc_train_x, vc_test_x)
    log('Get vc model input data done!')
    return vc_train_x, vc_train_y, vc_test_x, vc_test_y

def get_deploy_model_input_data(train):
    log('Get deploy model input data...')
    deploy_train_x, deploy_train_y= get_model_input_data(train=train,is_vc=0)
    log('Get deploy model input data done!')
    return deploy_train_x,deploy_train_y

def model_train(vc_train_x, vc_train_y, vc_test_x, vc_test_y,deploy_train_x,deploy_train_y):
    iterations_lgb, best_score_lgb, model_cv_lgb = lgb_cv(vc_train_x, vc_train_y, vc_test_x, vc_test_y, params_lgb, config_lgb['folds'], config_lgb['rounds'])
    model_lgb = lgb_deploy(deploy_train_x, deploy_train_y,iterations_lgb, params_lgb)
    log('Save model...')
    model_lgb.save_model('lgb-%f(r%d).txt' % (best_score_lgb, iterations_lgb))
    log('Model best score:' + str(best_score_lgb))
    log('Model best iteration:' + str(iterations_lgb))
    log('Save model done!')


def main():
    df=reload_all_feature()
    train=train_test_supplement_split(df)
    vc_train,vc_test=split_dataset_for_cv(train)
    vc_train_x, vc_train_y, vc_test_x, vc_test_y=get_vc_model_input_data(vc_train,vc_test)
    deploy_train_x,deploy_train_y=get_deploy_model_input_data(train)
    model_train(vc_train_x, vc_train_y, vc_test_x, vc_test_y,deploy_train_x,deploy_train_y)

if __name__ == "__main__":
    main()











