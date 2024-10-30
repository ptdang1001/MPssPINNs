import os
from multiprocessing import Pool

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('ignore')


def train_val_test(data, label, module_name, cur_step):
    if len(data) == 0:
        return {"module_name": module_name, "predictions": [], 'mean_mse': None, 'gene_importance': None}

    n_samples, n_features = data.shape
    #train_data = data.iloc[0:int(len(data) * 0.8), :]
    #train_label = label[0:int(len(data) * 0.8)]
    train_data, _, train_label, _ = train_test_split(data, label, test_size=0.2)


    # Train the model
    n_jobs = min(os.cpu_count(), n_features)
    num_round = 1000 + 500 * int(cur_step)
    model = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=num_round,
                          n_jobs=n_jobs,
                          reg_alpha=0.1, reg_lambda=0.1, random_state=42, silent=True, verbose=-1, max_depth=n_features)

    # train the model
    model.fit(train_data, train_label)

    # predict
    # test_data = lgb.Dataset(data, label=label)
    y_pred = model.predict(data, num_iteration=model.best_iteration_)
    y_pred = abs(y_pred)

    # get the mse
    mse = mean_squared_error(label, y_pred)
    # print("mse:", mse)

    # get the feature importance
    feature_importance = model.feature_importances_
    feature_importance = feature_importance / sum(feature_importance)

    # the name of the features
    feature_names = data.columns  # x is a Data frame
    importances = list(zip(feature_names, feature_importance))
    # print("importances:", importances)

    return {"module_name": module_name, "predictions": y_pred, 'mean_mse': mse, 'gene_importance': importances}


def check_flux_col_variation(flux, sampls_mean):
    flux = flux.multiply(sampls_mean, axis=0)
    return flux


def regression(samples_modules_mpo, samples_mean, modules_normalizedGenes, compounds_modules, cur_step, args):
    # the default regression method is random forest regression
    samples_modules_regression = {}
    samples_genes_importance = {}
    mse_steps = []
    param_list = [(modules_normalizedGenes[module_name], samples_modules_mpo[module_name].values, module_name, cur_step)
                  for module_name in samples_modules_mpo.columns.values]
    '''
    n_pool = min(os.cpu_count(), len(param_list))
    with Pool(n_pool) as pool:
        res = pool.starmap(train_val_test, param_list)
    '''
    res = [train_val_test(*param) for param in param_list]
    for cur_res in res:
        if len(cur_res['predictions']) == 0:
            continue
        samples_modules_regression[cur_res['module_name']] = cur_res['predictions']
        mse_steps.append(cur_res['mean_mse'])
        samples_genes_importance[cur_res['module_name']] = cur_res['gene_importance']

    samples_modules_regression = pd.DataFrame.from_dict(samples_modules_regression)
    samples_modules_regression.index = samples_modules_mpo.index
    no_genes_modules = list(set(samples_modules_mpo.columns) - set(samples_modules_regression.columns))
    if len(no_genes_modules) > 0:
        samples_modules_regression[no_genes_modules] = samples_modules_mpo[no_genes_modules]
    samples_modules_regression = samples_modules_regression[samples_modules_mpo.columns]

    # normalize the data by row
    # if samples_modules_regression.std(axis=0).mean() < 0.1:
    if False:
        samples_modules_regression = samples_modules_regression.div(samples_modules_regression.sum(axis=1), axis=0) * (
                10.0 ** (len(str(len(compounds_modules.columns)))))
        samples_modules_regression = check_flux_col_variation(samples_modules_regression, samples_mean)

    return samples_modules_regression, mse_steps, samples_genes_importance
