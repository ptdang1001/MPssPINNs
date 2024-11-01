# -*-coding:utf-8-*-

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

# my libs
from MPO_LoopOrAnchor.utils.data_interface import Factor_Graph
from MPO_LoopOrAnchor.utils.data_interface import get_imbalanceLoss
from MPO_LoopOrAnchor.utils.model_interface import MPO


def get_loss(compounds_modules, flux):
    flux = abs(flux)
    coefficient_of_variation = flux.std(axis=0) / flux.mean(axis=0)
    coefficient_of_variation = np.nan_to_num(coefficient_of_variation)

    # normalize the data by row
    flux = np.divide(flux, np.linalg.norm(flux, axis=1).reshape(-1, 1)) * (10.0**(len(str(len(compounds_modules.columns)))))
    imbalanceLoss_values = []
    for belief_i in flux:
        tmp1 = belief_i * compounds_modules
        tmp2 = np.sum(tmp1, axis=1)
        #tmp3 = tmp2 ** 2
        tmp3 = np.abs(tmp2)
        tmp4 = np.sum(tmp3)
        tmp5 = np.round(tmp4, 3)
        imbalanceLoss_values.append(tmp5)

    return {'mean_cv':np.mean(coefficient_of_variation),'mean_imbalanceLoss':np.mean(imbalanceLoss_values)}


def get_one_sample_flux(sample_name, factors_nodes, belief_old, factors, nodes, main_branch, args):
    mpo = MPO(factors_nodes, belief_old.copy(), factors, nodes, main_branch, args)

    # run the belief propagation
    mpo.run()

    belief_predicted_set = []
    for belief in mpo._belief_new_set:
        belief_predicted_set.append(belief[0])

    if len(belief_predicted_set) > 1:
        belief_predicted_set = np.stack(belief_predicted_set)
    # print("belief_predicted_set:\n{0}\n".format(belief_predicted_set))

    imbalanceLoss_values = get_imbalanceLoss(factors_nodes, belief_predicted_set)
    # print("\nall imbalanceLoss_values:{0}\n".format(imbalanceLoss_values))
    min_idx = imbalanceLoss_values.index(min(imbalanceLoss_values))
    if imbalanceLoss_values[min_idx] == imbalanceLoss_values[-1]:
        min_idx = len(imbalanceLoss_values) - 1
    # print("min_idx:{0}\n".format(min_idx))
    belief_predicted = belief_predicted_set[min_idx]
    return sample_name, belief_predicted


# @pysnooper.snoop()
def run_mpo(factors_nodes, samples_modules_input, main_branch, args):
    factor_graph = Factor_Graph(factors_nodes)  # This is a bipartite graph.
    samples_modules_mpo = {}

    res = []
    n_processes = min(os.cpu_count(), samples_modules_input.shape[0])
    pool = Pool(n_processes)
    for sample_i in range(samples_modules_input.shape[0]):
        sample_name = samples_modules_input.index.values[sample_i]
        belief_old = None
        belief_old = samples_modules_input.iloc[sample_i, :].values.tolist()
        belief_old = pd.DataFrame(belief_old)
        belief_old = belief_old.T
        belief_old.columns = samples_modules_input.columns
        res.append(pool.apply_async(func=get_one_sample_flux,
                                    args=(
                                        sample_name, factors_nodes, belief_old, factor_graph._factors,
                                        factor_graph._nodes, main_branch,
                                        args)))
    pool.close()
    pool.join()
    for res_i in res:
        sample_name, belief_predicted = res_i.get()
        samples_modules_mpo[sample_name] = belief_predicted

    samples_modules_mpo = pd.DataFrame.from_dict(samples_modules_mpo, orient='index')
    samples_modules_mpo.columns = samples_modules_input.columns
    sample_names = samples_modules_input.index.values
    samples_modules_mpo = samples_modules_mpo.loc[sample_names, :]
    return samples_modules_mpo


def check_flux_col_variation(flux, sampls_mean):
    flux = flux.multiply(sampls_mean, axis=0)
    return flux


# @pysnooper.snoop()
def mpo(compounds_modules, samples_modules_input, main_branch, samples_mean, args):
    samples_modules_mpo = None
    samples_modules_mpo = run_mpo(compounds_modules.copy(), samples_modules_input.copy(), main_branch, args)
    samples_modules_mpo = samples_modules_mpo.abs()
    #if samples_modules_mpo.std(axis=0).mean() < 0.1:
    if True:
        samples_modules_mpo = samples_modules_mpo.div(samples_modules_mpo.sum(axis=1), axis=0)* (10.0**(len(str(len(compounds_modules.columns)))))
        samples_modules_mpo = check_flux_col_variation(samples_modules_mpo, samples_mean)
    return samples_modules_mpo
