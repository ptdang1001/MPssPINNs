# -*-coding:utf-8-*-
import sys
import os
from datetime import datetime

# Third party libraries
import numpy as np
import pandas as pd

# import pysnooper
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functools import reduce
from collections import defaultdict, deque
import shutil

SEP_SIGN = "*" * 100


def get_output_path(args):
    network_name = args.network_name
    print("\n network_name:{0} \n".format(network_name))
    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp
    formatted_timestamp = current_timestamp.strftime("%Y%m%d%H%M%S")
    data_file_name = args.geneExpression_file_name.split(".csv")[0]
    folder_name = (
        f"{data_file_name}-{network_name}-{args.experiment_name}_{formatted_timestamp}"
    )
    # if folder already exists, add a number to the folder name
    if os.path.exists(f"{args.output_dir}{folder_name}/"):
        random_number = np.random.randint(1, 999)
        folder_name = f"{data_file_name}-{network_name}-{args.experiment_name}_{formatted_timestamp}_{str(random_number).zfill(3)}"
    output_dir = f"{args.output_dir}{folder_name}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir, folder_name


def pca_components_selection(geneExpression, modules_genes, n_components=0.9):
    geneExpression_pca = []
    X_scaler = StandardScaler()
    modules_genes_pca = {}

    for module_i, genes in modules_genes.items():
        print("\n cur genes:{0}-{1} \n".format(module_i, genes))

        genes_intersection = None
        genes_intersection = list(
            set(genes).intersection(set(list(geneExpression.columns.values)))
        )

        x = None
        x = geneExpression[genes_intersection]

        pca = PCA(n_components=n_components)
        x_scaled = X_scaler.fit_transform(x.values.copy())
        x_pca = pca.fit_transform(x_scaled)

        n_genes_pca = 0
        n_genes_pca = pca.n_components_
        genes_pca = [
            module_i + "_gene_" + str(i) + "_pca" for i in range(1, n_genes_pca + 1)
        ]
        modules_genes_pca[module_i] = genes_pca

        if len(geneExpression_pca) == 0:
            geneExpression_pca = pd.DataFrame(x_pca, columns=genes_pca)
        else:
            x_pca = pd.DataFrame(x_pca, columns=genes_pca)
            geneExpression_pca = pd.concat([geneExpression_pca, x_pca], axis=1)

    geneExpression_pca.index = geneExpression.index

    return geneExpression_pca, modules_genes_pca


def remove_non_connected_graph(compounds_flux):
    BDG = nx.DiGraph()

    for node_name in compounds_flux.columns.values:
        BDG.add_node(
            node_name,
            desc=node_name,
            color="red",
            shape="o",
            bipartite=1,
            # node_type="node",
        )

    for factor_name in compounds_flux.index.values:
        idx = np.where(compounds_flux.loc[factor_name, :] == 1)[0]
        parent_nodes = (
            list(np.take(compounds_flux.columns.values, idx)) if len(idx) else []
        )

        idx = np.where(compounds_flux.loc[factor_name, :] == -1)[0]
        child_nodes = (
            list(np.take(compounds_flux.columns.values, idx)) if len(idx) else []
        )

        BDG.add_node(
            factor_name,
            desc=factor_name,
            color="orange",
            shape="s",
            bipartite=0,
            # node_type="factor",
        )

        for parent_node_i in parent_nodes:
            BDG.add_edge(parent_node_i, factor_name)

        for child_node_i in child_nodes:
            BDG.add_edge(factor_name, child_node_i)
    BDG = BDG.to_undirected()
    largest_connected_component = max(nx.connected_components(BDG), key=len)

    keep_nodes = list(
        largest_connected_component.intersection(set(compounds_flux.columns.values))
    )
    keep_factors = list(
        largest_connected_component.intersection(set(compounds_flux.index.values))
    )
    compounds_flux = compounds_flux.loc[keep_factors, keep_nodes]
    return compounds_flux


def init_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)
    return True


def remove_allZero_rowAndCol(factors_nodes):
    # remove all zero rows and columns
    factors_nodes = factors_nodes.loc[~(factors_nodes == 0).all(axis=1), :]
    factors_nodes = factors_nodes.loc[:, ~(factors_nodes == 0).all(axis=0)]
    return factors_nodes


def remove_outside_compounds(factors_nodes):
    n_factors, n_nodes = factors_nodes.shape
    keep_idx = []
    print(SEP_SIGN)
    for i in range(n_factors):
        if (factors_nodes.iloc[i, :] >= 0).all() or (
            factors_nodes.iloc[i, :] <= 0
        ).all():
            print("Remove Compound:{0}".format(factors_nodes.index.values[i]))
            continue
        else:
            keep_idx.append(i)
    factors_nodes = factors_nodes.iloc[keep_idx, :]
    factors_nodes = remove_allZero_rowAndCol(factors_nodes)
    print(SEP_SIGN)
    # print(SEP_SIGN)
    # print("\n compounds_modules sample:\n {0} \n".format(factors_nodes))
    # print(SEP_SIGN)
    return factors_nodes


def update_compoundsModules_modulesGenes(
    compounds_modules, modules_genes, geneExpression
):
    compounds_modules = remove_non_connected_graph(compounds_modules)

    all_modules = list(
        set(list(modules_genes.keys())).intersection(
            set(compounds_modules.columns.values.tolist())
        )
    )
    all_modules.sort()
    compounds_modules = compounds_modules[all_modules]
    compounds_modules = remove_non_connected_graph(compounds_modules)
    compounds_modules = remove_outside_compounds(compounds_modules)

    all_modules = list(
        set(list(modules_genes.keys())).intersection(
            set(compounds_modules.columns.values.tolist())
        )
    )
    all_modules.sort()
    compounds_modules = compounds_modules[all_modules]

    modules_genes_new = {}
    genes_new = []
    for module in all_modules:
        modules_genes_new[module] = modules_genes[module]
        genes_new.extend(modules_genes[module])
    modules_genes = modules_genes_new

    genes_new = list(
        set(genes_new).intersection(set(geneExpression.columns.values.tolist()))
    )
    geneExpression = geneExpression[genes_new]

    # update genes in modules_genes
    modules_genes_new_new = {}
    for module_i in modules_genes_new.keys():
        cur_genes = modules_genes_new[module_i]
        cur_genes = list(set(cur_genes).intersection(set(genes_new)))
        if len(cur_genes) == 0:
            continue
        modules_genes_new_new[module_i] = cur_genes

    # update modules in compounds_modules
    all_modules = list(
        set(list(modules_genes_new_new.keys())).intersection(
            set(compounds_modules.columns.values.tolist())
        )
    )
    all_modules.sort()
    compounds_modules = compounds_modules[all_modules]

    return compounds_modules, modules_genes_new_new, geneExpression


def perturb_modules(samples_modules, module_info, sm_id):
    times = 5.0
    if "SM_id" not in module_info.columns.values:
        print("Random perturb modules!")
        rdm_idx = np.random.choice(
            samples_modules.shape[1], int(samples_modules.shape[1] / 2)
        )[0]
        samples_modules.iloc[:, rdm_idx] = samples_modules.iloc[:, rdm_idx] * times
        return samples_modules

    print("SM_id moduels perturb!")
    # module_info['SM_id'] = module_info['SM_id'].astype(int)
    modules = module_info[module_info["SM_id"] == str(sm_id)].index.values
    if len(modules) == 0:
        modules = module_info[module_info["SM_id"] == sm_id].index.values
    print(modules)

    modules_intersection = list(
        set(modules).intersection(set(samples_modules.columns.values.tolist()))
    )
    modules_intersection = list(set(modules_intersection))
    modules_intersection.sort()
    print(modules_intersection)

    samples_modules[modules_intersection] = (
        samples_modules[modules_intersection] * times
    )
    return samples_modules


def load_module(args):
    read_path = os.path.join(args.network_dir, args.compounds_modules_file_name)
    args.network_name = args.compounds_modules_file_name.replace("_cmMat.csv", "")
    factors_nodes = pd.read_csv(read_path, index_col=0)
    factors_nodes.index = factors_nodes.index.map(lambda x: str(x))
    factors_nodes = factors_nodes.astype(int)

    # remove all zero rows and cols
    # factors_nodes = remove_allZero_rowAndCol(factors_nodes)
    # new_indexs = ['compound_' + str(i + 1) for i in range(factors_nodes.shape[0])]
    # factors_nodes.index = new_indexs
    """
    print(SEP_SIGN)
    print("\nCompoundss:{0}\n".format(factors_nodes.index.values))
    print("\nReactions:{0}\n".format(factors_nodes.columns.values))
    print("\nCompounds_Reactions shape:{0}\n".format(factors_nodes.shape))
    print("\n compounds_Reactions sample:\n {0} \n".format(factors_nodes))
    print(SEP_SIGN)
    """
    return factors_nodes, args


def load_geneExpression(args):
    read_path = os.path.join(args.input_dir, args.geneExpression_file_name)
    geneExpression = None

    if read_path.endswith(".csv.gz"):
        geneExpression = pd.read_csv(read_path, index_col=0, compression="gzip")
    elif read_path.endswith(".csv"):
        geneExpression = pd.read_csv(read_path, index_col=0)
    else:
        print("Wrong Gene Expression File Name!")
        return False

    # replace the nan with zero
    geneExpression = geneExpression.fillna(0.0)

    # remove the rows which are all zero
    geneExpression = geneExpression.loc[~(geneExpression == 0).all(axis=1), :]

    # remove duplicated samples
    geneExpression = geneExpression[~geneExpression.index.duplicated(keep="first")]

    # remove duplicated genes
    geneExpression = geneExpression.loc[
        :, ~geneExpression.columns.duplicated(keep="first")
    ]

    if "simulated_geneExpression" not in read_path:
        geneExpression = geneExpression.T

    print(SEP_SIGN)
    print("\n Gene Expression sample:\n {0} \n".format(geneExpression))
    print(SEP_SIGN)

    return geneExpression


def add_noise_to_geneExpression(geneExpression):
    noise = 0
    geneExpression = geneExpression + noise
    n_samples, n_genes = geneExpression.shape
    rdm_row_idx = np.random.choice(n_samples, int(0.2 * n_samples), replace=False)
    rdm_row_idx.sort()
    rdm_col_idx = np.random.choice(n_genes, int(0.2 * n_genes), replace=False)
    rdm_col_idx.sort()
    geneExpression.iloc[rdm_row_idx, rdm_col_idx] = 0.0
    return geneExpression.abs()


def generate_noise(n_samples, n_genes):
    noise = np.random.randn(n_samples, n_genes)
    noise = pd.DataFrame(noise)
    return noise.abs()


def load_modulesGenes(args):
    import json

    read_path = os.path.join(args.network_dir, args.modules_genes_file_name)
    # Opening JSON file
    f = open(read_path)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()

    print(SEP_SIGN)
    print("\n Reactions and contained genes:\n {0} \n".format(data))
    print(SEP_SIGN)

    return data


def intersect_samples_genes(geneExpression, modules_genes, compounds_modules, scaler):
    all_module_genes = []
    all_geneExpr_genes = set(geneExpression.columns.values.tolist())
    module_genes_new = {}
    no_genes_modules = []
    all_modules_in_compounds_modules = set(compounds_modules.columns.values.tolist())
    no_genes_modules.extend(
        list(
            set(compounds_modules.columns.values.tolist()).difference(
                set(modules_genes.keys())
            )
        )
    )

    print(SEP_SIGN)
    print("\n After Intersection! \n")
    modules_normalizedGenes = {module_i: {} for module_i in module_genes_new.keys()}
    # get the intersection of genes bewteen geneExpression and modules_genes
    for module_i, genes in modules_genes.items():
        if module_i not in all_modules_in_compounds_modules:
            continue

        cur_intersect_genes = list(set(set(genes).intersection(all_geneExpr_genes)))
        if len(cur_intersect_genes) != 0:
            module_genes_new[module_i] = cur_intersect_genes
            if scaler == None:
                modules_normalizedGenes[module_i] = geneExpression[cur_intersect_genes]
            else:
                tmp = scaler.fit_transform(
                    geneExpression[cur_intersect_genes].values.copy()
                )
                modules_normalizedGenes[module_i] = pd.DataFrame(
                    tmp, index=geneExpression.index, columns=cur_intersect_genes
                )
        else:
            modules_normalizedGenes[module_i] = []
            no_genes_modules.append(module_i)
        print(
            "\n Cur module_i:{0}, intersection of genes:{1}".format(
                module_i, cur_intersect_genes
            )
        )
        all_module_genes.extend(cur_intersect_genes)

    all_module_genes = list(set(all_module_genes))
    if len(all_module_genes) == 0:
        print("No Intersection of Genes!!")
        return [], [], [], [], [], []

    # update genes in geneExpression
    geneExpression = geneExpression[all_module_genes]

    samples_mean = geneExpression.mean(axis=1).fillna(1.0).replace(0.0, 1.0)
    samples_mean = (
        ((samples_mean - samples_mean.min()) / samples_mean.max()) + 1.987
    ) * 1.0

    return (
        geneExpression,
        module_genes_new,
        compounds_modules,
        samples_mean,
        modules_normalizedGenes,
        no_genes_modules,
    )


def check_intersect_genes(geneExpression_genes, modules_genes):
    geneExpression_genes = set(geneExpression_genes)
    modules_genes_new = {}
    for module_i, genes in modules_genes.items():
        cur_genes_intersection = ()
        cur_genes_intersection = set(genes).intersection(geneExpression_genes)
        if len(cur_genes_intersection) != 0:
            modules_genes_new[module_i] = list(cur_genes_intersection)
        print(
            "\n Cur module_i:{0}, intersection of genes:{1}".format(
                module_i, cur_genes_intersection
            )
        )

    return modules_genes_new


def shuffle_geneExpression(geneExpression, by="full_random"):
    n_row, n_col = geneExpression.shape
    if by == "full_random":
        n_total = n_row * n_col
        tmp_values = geneExpression.values
        tmp_values = tmp_values.reshape(n_total)
        rdm_idxs = np.random.choice(n_total, n_total, replace=False)
        tmp_values = tmp_values[rdm_idxs]
        tmp_values = tmp_values.reshape(n_row, n_col)
        tmp_values = pd.DataFrame(tmp_values)
        tmp_values.index = geneExpression.index
        tmp_values.columns = geneExpression.columns
        geneExpression = tmp_values

    return geneExpression


def save_samples_modules(samples_modules, method_name, cur_step, args):
    # samples_modules = min_max_normalization(samples_modules, by="col")

    save_dir = args.output_dir + "tmp_flux_res/" + method_name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = save_dir + "flux_" + method_name + "_" + str(cur_step).zfill(4) + ".csv"
    samples_modules.to_csv(save_path, index=True, header=True)


def load_samples_modules(args):
    read_path = args.output_dir + args.samples_modules_file_name
    samples_modules = pd.read_csv(read_path, index_col=0)
    return samples_modules


def remove_grad_files(save_step, args):
    end_str = "_grad_" + str(save_step).zfill(4) + ".csv"
    for file_i in os.listdir(args.output_dir):
        if "_grad_" in file_i:
            if not file_i.endswith(end_str):
                remove_path = args.output_dir + file_i
                os.remove(remove_path)


def save_snn_model_weights(save_step, args):
    target_folder = "Epoch_" + str(save_step).zfill(4)
    source_dir = (
        args.output_dir + "SNN/model_weights_checkpoints/" + target_folder + "/*"
    )
    target_dir = args.output_dir + "SNN/final_model_weights/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    command = "cp -r " + source_dir + " " + target_dir
    os.system(command)


def save_grad_file(args):
    files = [file_i for file_i in os.listdir(args.output_dir) if "_grad_" in file_i]
    grad_all = []
    for file_i in files:
        cur_module = "_".join(file_i.split("_")[2:4])
        cur_module_grad_df = pd.read_csv(args.output_dir + file_i, index_col=0)
        cur_module_grad_df.columns = cur_module_grad_df.columns.map(
            lambda gene_i: cur_module + "_" + gene_i
        )
        if len(grad_all) == 0:
            grad_all = cur_module_grad_df.copy()
        else:
            grad_all = pd.concat(
                [grad_all, cur_module_grad_df], axis=1, ignore_index=False
            )
        os.remove(args.output_dir + file_i)

    """
    if len(grad_all)== 0:
        print("No Gradients Found!!")
        return False
    """
    module_genes = grad_all.columns.values.tolist()
    module_genes.sort()
    grad_all = grad_all[module_genes]
    grad_all = grad_all.T
    grad_all.to_csv(args.output_dir + "flux_snn_grad.csv", index=True, header=True)

    return True


def min_max_normalization(matrix, by="col"):
    scaler = MinMaxScaler()

    if by == "col":
        matrix_col_scaled = pd.DataFrame(
            scaler.fit_transform(matrix), index=matrix.index, columns=matrix.columns
        )
        matrix_col_scaled[matrix_col_scaled < float("-inf")] = 0.0
        matrix_col_scaled[matrix_col_scaled == 0.0] = 100000.0
        matrix_col_scaled[matrix_col_scaled == 100000.0] = matrix_col_scaled.min().min()
        return matrix_col_scaled
    elif by == "row":
        matrix_row_scaled = pd.DataFrame(
            scaler.fit_transform(matrix.T),
            index=matrix.T.index,
            columns=matrix.T.columns,
        )
        matrix_row_scaled = matrix_row_scaled.T
        matrix_row_scaled[matrix_row_scaled < float("-inf")] = 0.0
        matrix_row_scaled[matrix_row_scaled == 0.0] = 1000000.0
        matrix_row_scaled[matrix_row_scaled == 1000000.0] = (
            matrix_row_scaled.min().min()
        )
        return matrix_row_scaled
    else:
        return "Error!!!"


def get_metabolism(compounds_modules, flux):
    samples_compounds = []
    for flux_i in flux.values:
        tmp1 = flux_i * compounds_modules.values
        tmp2 = np.sum(tmp1, axis=1)
        tmp3 = abs(tmp2)
        samples_compounds.append(tmp3.tolist())
    samples_compounds = pd.DataFrame(samples_compounds)
    samples_compounds.index = flux.index
    samples_compounds.columns = compounds_modules.index
    return samples_compounds


def get_imbalance_cv_loss(factors_nodes, flux):
    col_mean_flux = flux.mean(axis=0)
    col_mean_flux[col_mean_flux == 0] = 1.0
    coefficient_of_variation = flux.std(axis=0) / col_mean_flux
    coefficient_of_variation = np.nan_to_num(coefficient_of_variation)

    # normalize the data by row
    # flux = np.divide(flux, np.linalg.norm(flux, axis=1).reshape(-1, 1)) * (
    #            10.0 ** (len(str(len(factors_nodes.columns)))))
    flux = np.divide(flux, np.linalg.norm(flux, axis=1).reshape(-1, 1))
    imbalanceLoss_values = []
    for belief_i in flux:
        tmp1 = belief_i * factors_nodes
        tmp2 = np.sum(tmp1, axis=1)
        # tmp3 = tmp2 ** 2
        tmp3 = np.abs(tmp2)
        tmp4 = np.sum(tmp3)
        tmp5 = np.round(tmp4, 3)
        imbalanceLoss_values.append(tmp5)
    # print(f"coefficient_of_variation: {coefficient_of_variation}")
    # print(f"imbalanceLoss_values: {imbalanceLoss_values}")
    return {
        "mean_cv": np.mean(coefficient_of_variation),
        "mean_imbalance": np.mean(imbalanceLoss_values),
    }


# @pysnooper.snoop()
def get_imbalanceLoss(factors_nodes, belief_set):
    belief_set = pd.DataFrame(belief_set)
    # belief_set = belief_set.div(belief_set.sum(axis=1), axis=0) * 100
    belief_set = belief_set.div(np.linalg.norm(belief_set, axis=1), axis=0) * (
        10.0 ** (len(str(len(factors_nodes.columns))))
    )
    imbalanceLoss_values = []
    for belief_i in belief_set.values:
        tmp1 = belief_i * factors_nodes.values
        tmp2 = np.sum(tmp1, axis=1)
        # tmp3 = tmp2 ** 2
        tmp3 = np.abs(tmp2)
        tmp4 = np.sum(tmp3)
        tmp5 = np.round(tmp4, 3)
        imbalanceLoss_values.append(tmp5)

    return imbalanceLoss_values


def get_mse(matrix1, matrix2, by="col"):
    mse = None
    if by == "col":
        mse = np.mean((matrix1.values - matrix2.values) ** 2, axis=0)
    if by == "row":
        mse = np.mean((matrix1.values - matrix2.values) ** 2, axis=1)
    return mse


def get_std_scale_imbalanceLoss_realData(
    compounds_modules,
    samples_modules_scfea,
    samples_modules_mpo,
    samples_modules_snn,
    samples_modules_mposnn,
):
    """
    samples_modules_scfea = min_max_normalization(samples_modules_scfea, by='col')
    samples_modules_mpo = min_max_normalization(samples_modules_mpo, by='col')
    samples_modules_snn = min_max_normalization(samples_modules_snn, by='col')
    """
    times = 1000.0
    samples_modules_scfea = (
        samples_modules_scfea.div(samples_modules_scfea.sum(axis=1), axis=0) * times
    )
    samples_modules_mpo = (
        samples_modules_mpo.div(samples_modules_mpo.sum(axis=1), axis=0) * times
    )
    samples_modules_snn = (
        samples_modules_snn.div(samples_modules_snn.sum(axis=1), axis=0) * times
    )
    samples_modules_mposnn = (
        samples_modules_mposnn.div(samples_modules_mposnn.sum(axis=1), axis=0) * times
    )

    scale_mean_all_scfea = samples_modules_scfea.mean().mean()
    scale_mean_all_mpo = samples_modules_mpo.mean().mean()
    scale_mean_all_snn = samples_modules_snn.mean().mean()
    scale_mean_all_mposnn = samples_modules_mposnn.mean().mean()

    imbalanceLoss_mean_scfea = np.mean(
        get_imbalanceLoss(compounds_modules, samples_modules_scfea.values)
    )
    imbalanceLoss_mean_mpo = np.mean(
        get_imbalanceLoss(compounds_modules, samples_modules_mpo.values)
    )
    imbalanceLoss_mean_snn = np.mean(
        get_imbalanceLoss(compounds_modules, samples_modules_snn.values)
    )
    imbalanceLoss_mean_mposnn = np.mean(
        get_imbalanceLoss(compounds_modules, samples_modules_mposnn.values)
    )

    # normalize the data by col
    # samples_modules_snn = samples_modules_snn.T.div(samples_modules_snn.max(axis=0), axis=0).T
    # samples_modules_scfea = samples_modules_scfea.T.div(samples_modules_scfea.max(axis=0), axis=0).T
    # samples_modules_mpo = samples_modules_mpo.T.div(samples_modules_mpo.max(axis=0), axis=0).T

    std_mean_col_scfea = np.mean(np.std(samples_modules_scfea, axis=0))
    std_mean_col_mpo = np.mean(np.std(samples_modules_mpo, axis=0))
    std_mean_col_snn = np.mean(np.std(samples_modules_snn, axis=0))
    std_mean_col_mposnn = np.mean(np.std(samples_modules_mposnn, axis=0))

    res = {
        "std_mean_col_scfea": std_mean_col_scfea,
        "std_mean_col_mpo": std_mean_col_mpo,
        "std_mean_col_snn": std_mean_col_snn,
        "std_mean_col_mposnn": std_mean_col_mposnn,
        "imbalanceLoss_mean_scfea": imbalanceLoss_mean_scfea,
        "imbalanceLoss_mean_mpo": imbalanceLoss_mean_mpo,
        "imbalanceLoss_mean_snn": imbalanceLoss_mean_snn,
        "imbalanceLoss_mean_mposnn": imbalanceLoss_mean_mposnn,
        "scale_mean_all_scfea": scale_mean_all_scfea,
        "scale_mean_all_mpo": scale_mean_all_mpo,
        "scale_mean_all_snn": scale_mean_all_snn,
        "scale_mean_all_mposnn": scale_mean_all_mposnn,
    }

    return res


def plot_FactorGraph(compounds_flux, title_name, save_path):
    BDG = nx.DiGraph()

    for node_name in compounds_flux.columns.values:
        BDG.add_node(
            node_name,
            desc=node_name,
            color="red",
            shape="o",
            bipartite=1,
            # node_type="node",
        )

    for factor_name in compounds_flux.index.values:
        idx = np.where(compounds_flux.loc[factor_name, :] == 1)[0]
        parent_nodes = (
            list(np.take(compounds_flux.columns.values, idx)) if len(idx) else []
        )

        idx = np.where(compounds_flux.loc[factor_name, :] == -1)[0]
        child_nodes = (
            list(np.take(compounds_flux.columns.values, idx)) if len(idx) else []
        )

        BDG.add_node(
            factor_name,
            desc=factor_name,
            color="orange",
            shape="s",
            bipartite=0,
            # node_type="factor",
        )

        for parent_node_i in parent_nodes:
            BDG.add_edge(parent_node_i, factor_name)

        for child_node_i in child_nodes:
            BDG.add_edge(factor_name, child_node_i)

    # colors = [BDG.nodes[i]['color'] for i in BDG.nodes()]
    # shapes = [BDG.nodes[i]['shape'] for i in BDG.nodes()]

    figure(figsize=(30, 40), dpi=400)

    # draw graph with labels
    pos = nx.nx_pydot.graphviz_layout(BDG)
    node_size = 1000
    nx.draw(BDG, pos, with_labels=True)
    nx.draw_networkx_nodes(
        BDG,
        pos,
        nodelist=compounds_flux.columns.values,
        node_color="yellow",
        node_shape="o",
        node_size=node_size,
    )
    nx.draw_networkx_nodes(
        BDG,
        pos,
        nodelist=compounds_flux.index.values,
        node_color="lightblue",
        node_shape="s",
        node_size=node_size,
    )
    nx.draw_networkx_edges(BDG, pos, width=3.0, arrowsize=40)
    plt.title(title_name)
    # plt.show()
    plt.savefig(save_path)

    return True


def get_cycles(BDG):
    cycles_in_graph = list(nx.simple_cycles(BDG))
    cycles_in_graph.sort(key=lambda x: -len(x))
    print(SEP_SIGN)
    print("\n Cycles in BDG:\n {0} \n".format(cycles_in_graph))
    print(SEP_SIGN)
    return cycles_in_graph


# @pysnooper.snoop()
def pre_process_cycle(
    DG, nodes_in_cycle, nodes_in_cycle_set, factors_in_cycle, factors_in_cycle_set
):
    factor_parentNode_childNode_outsideCycle = defaultdict(dict)
    startAnchorNode_factors_nextAnchorNode = defaultdict(dict)

    drop_nodes = []
    drop_factors = []
    cycle_valid_flag = False

    #################################################################################################################################
    # no anchor nodes in cycle
    if (
        DG.nodes[nodes_in_cycle[0]]["n_parent"] + DG.nodes[nodes_in_cycle[0]]["n_child"]
        <= 2
    ):
        drop_nodes = nodes_in_cycle
        drop_factors = factors_in_cycle

        # get outside interaction factors in cycle
        outside_in_node = False
        outside_out_node = False
        for factor_i in factors_in_cycle:
            if DG.nodes[factor_i]["n_parent"] + DG.nodes[factor_i]["n_child"] <= 2:
                continue

            outsideParentNode = (
                list(
                    filter(
                        lambda x: (x not in nodes_in_cycle_set),
                        list(DG.predecessors(factor_i)),
                    )
                )
                + []
            )
            if len(outsideParentNode) >= 1:
                outside_in_node = True
            factor_parentNode_childNode_outsideCycle[factor_i][
                "outsideParentNode"
            ] = outsideParentNode

            outsideChildNode = (
                list(
                    filter(
                        lambda x: (x not in nodes_in_cycle_set),
                        list(DG.successors(factor_i)),
                    )
                )
                + []
            )
            if len(outsideChildNode) >= 1:
                outside_out_node = True
            factor_parentNode_childNode_outsideCycle[factor_i][
                "outsideChildNode"
            ] = outsideChildNode

        if (outside_in_node and outside_out_node) or (
            not outside_in_node and not outside_out_node
        ):
            cycle_valid_flag = True

        return (
            cycle_valid_flag,
            drop_nodes,
            drop_factors,
            factor_parentNode_childNode_outsideCycle,
            startAnchorNode_factors_nextAnchorNode,
        )

    #################################################################################################################################
    # there is only one anchor node in cycle
    if (
        DG.nodes[nodes_in_cycle[1]]["n_parent"] + DG.nodes[nodes_in_cycle[1]]["n_child"]
        <= 2
    ):

        start_anchor_node = nodes_in_cycle[0]

        # startAnchorNode_factors_nextAnchorNode[start_anchor_node]["next_drop_factors"] = []
        startAnchorNode_factors_nextAnchorNode[start_anchor_node][
            "next_anchor_node"
        ] = start_anchor_node

        # get outside interaction factors in cycle
        outside_in_node = False
        outside_out_node = False
        for factor_i in factors_in_cycle:
            if DG.nodes[factor_i]["n_parent"] + DG.nodes[factor_i]["n_child"] <= 2:
                continue

            outsideParentNode = (
                list(
                    filter(
                        lambda x: (x not in nodes_in_cycle_set),
                        list(DG.predecessors(factor_i)),
                    )
                )
                + []
            )
            if not outside_in_node and len(outsideParentNode) >= 1:
                outside_in_node = True
            factor_parentNode_childNode_outsideCycle[factor_i][
                "outsideParentNode"
            ] = outsideParentNode
            outsideChildNode = (
                list(
                    filter(
                        lambda x: (x not in nodes_in_cycle_set),
                        list(DG.successors(factor_i)),
                    )
                )
                + []
            )
            if not outside_out_node and len(outsideChildNode) >= 1:
                outside_out_node = True
            factor_parentNode_childNode_outsideCycle[factor_i][
                "outsideChildNode"
            ] = outsideChildNode

        startAnchorNode_factors_nextAnchorNode[start_anchor_node][
            "next_outIteraction_factors"
        ] = list(factor_parentNode_childNode_outsideCycle.keys())

        if (outside_in_node and outside_out_node) or (
            not outside_in_node and not outside_out_node
        ):
            cycle_valid_flag = True

        # if there are out outInteraction_factors
        if len(factor_parentNode_childNode_outsideCycle) > 0:
            drop_nodes = nodes_in_cycle[1:]
            drop_factors = factors_in_cycle

        return (
            cycle_valid_flag,
            drop_nodes,
            drop_factors,
            factor_parentNode_childNode_outsideCycle,
            startAnchorNode_factors_nextAnchorNode,
        )

    #################################################################################################################################
    # there are at least 2 anchor nodes in the cycle

    outInteraction_factors = []
    next_drop_factors = []
    start_anchor_node = nodes_in_cycle.copy()[0]
    # startAnchorNode_factors_endAnchorNode[start_anchor_node]['next_factors'] = []
    # startAnchorNode_factors_endAnchorNode[start_anchor_node]['next_anchor_node'] =

    cur_anchor_node = nodes_in_cycle.copy()[0]
    # next_anchor_node=find_next_anchor_node(DG,cur_anchor_node,nodes_in_cycle_set,factors_in_cycle_set)
    next_factor = list(
        filter(
            lambda x: (x in factors_in_cycle_set), list(DG.successors(cur_anchor_node))
        )
    )[0]
    next_node = list(
        filter(lambda x: (x in nodes_in_cycle_set), list(DG.successors(next_factor)))
    )[0]

    while True:

        if DG.nodes[next_factor]["n_parent"] + DG.nodes[next_factor]["n_child"] > 2:
            outInteraction_factors.append(next_factor)

            outsideParentNode = (
                list(
                    filter(
                        lambda x: (x not in nodes_in_cycle_set),
                        DG.predecessors(next_factor),
                    )
                )
                + []
            )
            if len(outsideParentNode) >= 1:
                outside_in_node = True
            factor_parentNode_childNode_outsideCycle[next_factor][
                "outsideParentNode"
            ] = outsideParentNode

            outsideChildNode = (
                list(
                    filter(
                        lambda x: (x not in nodes_in_cycle_set),
                        DG.successors(next_factor),
                    )
                )
                + []
            )
            if len(outsideChildNode) >= 1:
                outside_out_node = True
            factor_parentNode_childNode_outsideCycle[next_factor][
                "outsideChildNode"
            ] = outsideChildNode

        else:
            next_drop_factors.append(next_factor)
            drop_factors.append(next_factor)

        next_factor = list(
            filter(lambda x: (x in factors_in_cycle_set), DG.successors(next_node))
        )[0]

        if DG.nodes[next_node]["n_parent"] + DG.nodes[next_node]["n_child"] > 2:
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                "next_outInteraction_factors"
            ] = outInteraction_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                "next_drop_factors"
            ] = next_drop_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                "next_anchor_node"
            ] = next_node
            cur_anchor_node = next_node
            outInteraction_factors = []
            next_drop_factors = []
        else:
            drop_nodes.append(next_node)

        next_node = list(
            filter(lambda x: (x in nodes_in_cycle_set), DG.successors(next_factor))
        )[0]
        if next_node == start_anchor_node:
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                "next_outInteraction_factors"
            ] = outInteraction_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                "next_drop_factors"
            ] = next_drop_factors.copy()
            startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                "next_anchor_node"
            ] = start_anchor_node
            break

    if (not cycle_valid_flag) and (
        (outside_in_node and outside_out_node)
        or (not outside_in_node and not outside_out_node)
    ):
        cycle_valid_flag = True

    return (
        cycle_valid_flag,
        drop_nodes,
        drop_factors,
        factor_parentNode_childNode_outsideCycle,
        startAnchorNode_factors_nextAnchorNode,
    )


def find_start_anchor_node(startAnchorNode_factors_nextAnchorNode):
    for anchor_node_i in startAnchorNode_factors_nextAnchorNode.keys():
        next_outInteraction_factors = startAnchorNode_factors_nextAnchorNode[
            anchor_node_i
        ]["next_outInteraction_factors"]
        next_anchor_node = startAnchorNode_factors_nextAnchorNode[anchor_node_i][
            "next_anchor_node"
        ]

        if len(next_outInteraction_factors) == 0 and next_anchor_node:
            return next_anchor_node

    return list(startAnchorNode_factors_nextAnchorNode.keys())[0]


# @pysnooper.snoop()
def find_otherNodes_affectedBy_anchorNodes(DG, anchor_node, by="BFS"):
    q = deque()
    visited = set()
    q.append(anchor_node)
    # visited.add(anchor_node)
    all_affected_nodes = []

    while q:
        for _ in range(len(q)):
            cur = q.popleft()

            if cur not in visited and DG.nodes[cur]["type"] == "node":
                visited.add(cur)

                parent_child_factors = list(DG.predecessors(cur)) + list(
                    DG.successors(cur)
                )
                parent_child_factors = list(
                    filter(lambda x: (x not in visited), parent_child_factors)
                )
                if len(parent_child_factors) == 0:
                    continue

                for factor_i in parent_child_factors:
                    visited.add(factor_i)

                    parent_child_nodes = list(DG.predecessors(factor_i)) + list(
                        DG.successors(factor_i)
                    )
                    if len(parent_child_nodes) > 2:
                        continue

                    affected_nodes = list(
                        filter(lambda x: (x not in visited), parent_child_nodes)
                    )
                    all_affected_nodes.extend(affected_nodes)
                    q.extend(affected_nodes)
                    # visited=visited.union(set(affected_nodes))

                continue

            # if not visited(cur) and DG.nodes[cur]['type']=='factor':
            #    continue

    return all_affected_nodes


# @pysnooper.snoop()
def collapse_cycles(factors_nodes, DG, cycles_in_graph):
    # factors = set(factors_nodes.index.values)
    # nodes = set(factors_nodes.columns.values)

    # collapse the cycle
    all_drop_factors = set()
    all_drop_nodes = set()
    visited_outside_nodes = set()

    for i in range(len(cycles_in_graph)):
        factorsAndNodes_in_cycle = cycles_in_graph[i]
        nodes_in_cycle = list(
            filter(lambda x: DG.nodes[x]["type"] == "node", factorsAndNodes_in_cycle)
        )
        nodes_in_cycle.sort(
            key=lambda x: -(DG.nodes[x]["n_parent"] + DG.nodes[x]["n_child"])
        )
        nodes_in_cycle_set = set(nodes_in_cycle)

        factors_in_cycle = list(
            filter(lambda x: DG.nodes[x]["type"] == "factor", factorsAndNodes_in_cycle)
        )
        factors_in_cycle_set = set(factors_in_cycle)

        (
            cycle_valid_flag,
            drop_nodes,
            drop_factors,
            factor_parentNode_childNode_outsideCycle,
            startAnchorNode_factors_nextAnchorNode,
        ) = pre_process_cycle(
            DG,
            nodes_in_cycle,
            nodes_in_cycle_set,
            factors_in_cycle,
            factors_in_cycle_set,
        )

        #################################################################################################################################
        # the cycle is valid
        if not cycle_valid_flag:
            print(
                "\nThe cycle is not valid!!\n There is only In node or Out node linked to factors in the cycle!\n"
            )
            return

        ##################################################################################################################################
        # no anchor node in the cycle
        # merger all anchor factors into one collapse_factor and link all in and out nodes to this collapse_factor
        # remove all nodes and factors in cycle
        if len(startAnchorNode_factors_nextAnchorNode) == 0:
            all_drop_factors = all_drop_factors.union(set(drop_factors))
            all_drop_nodes = all_drop_nodes.union(set(drop_nodes))

            collapse_factor = "collapse_factor_" + "_".join(
                list(factor_parentNode_childNode_outsideCycle.keys())
            )
            factors_nodes.loc[collapse_factor, :] = 0
            for interaction_factor_i in factor_parentNode_childNode_outsideCycle.keys():
                parent_nodes = factor_parentNode_childNode_outsideCycle[
                    interaction_factor_i
                ]["outsideParentNode"]
                child_nodes = factor_parentNode_childNode_outsideCycle[
                    interaction_factor_i
                ]["outsideChildNode"]
                if parent_nodes:
                    for parent_node_i in parent_nodes:
                        if parent_node_i in visited_outside_nodes:
                            continue
                        factors_nodes.loc[collapse_factor, parent_node_i] = 1
                        visited_outside_nodes.add(parent_node_i)

                if child_nodes:
                    for child_node_i in child_nodes:
                        if child_node_i in visited_outside_nodes:
                            continue
                        factors_nodes.loc[collapse_factor, child_node_i] = -1
                        visited_outside_nodes.add(child_node_i)

            continue

        #################################################################################################################################
        # only one anchor node in cycle
        if len(startAnchorNode_factors_nextAnchorNode.keys()) == 1:
            start_anchor_node = None
            start_anchor_node = list(
                startAnchorNode_factors_nextAnchorNode.keys()
            ).copy()[0]

            # no in node and no out node linked to the factors in the cycle
            if len(factor_parentNode_childNode_outsideCycle) == 0:
                print("There are no in and out interaction factors in the cycle!")
                # break the edge from the anchor node and its in cycle parent factor

                inCycle_parentFactor = list(
                    filter(
                        lambda x: x in factors_in_cycle_set,
                        list(DG.predecessors(start_anchor_node)),
                    )
                ).copy()[0]
                factors_nodes.loc[inCycle_parentFactor, start_anchor_node] = 0

            else:
                all_drop_factors = all_drop_factors.union(set(drop_factors))
                all_drop_nodes = all_drop_nodes.union(set(drop_nodes))

                # There are in edge and out edge linked to the factors in the cycle
                # start_anchor_node = next(iter(startAnchorNode_factors_nextAnchorNode))

                collapse_factor = "collapse_factor_" + "_".join(
                    list(factor_parentNode_childNode_outsideCycle.keys())
                )
                factors_nodes.loc[collapse_factor, :] = 0
                for (
                    interaction_factor_i
                ) in factor_parentNode_childNode_outsideCycle.keys():
                    parent_nodes = factor_parentNode_childNode_outsideCycle[
                        interaction_factor_i
                    ]["outsideParentNode"]
                    child_nodes = factor_parentNode_childNode_outsideCycle[
                        interaction_factor_i
                    ]["outsideChildNode"]
                    if parent_nodes:
                        for parent_node_i in parent_nodes:
                            if parent_node_i in visited_outside_nodes:
                                continue
                            factors_nodes.loc[collapse_factor, parent_node_i] = 1
                            visited_outside_nodes.add(parent_node_i)

                    if child_nodes:
                        for child_node_i in child_nodes:
                            if child_node_i in visited_outside_nodes:
                                continue
                            factors_nodes.loc[collapse_factor, child_node_i] = -1
                            visited_outside_nodes.add(child_node_i)

                continue

        #################################################################################################################################
        # more than one anchor node in cycle
        if len(startAnchorNode_factors_nextAnchorNode.keys()) > 1:
            all_drop_nodes = all_drop_nodes.union(set(drop_nodes))
            all_drop_factors = all_drop_factors.union(set(drop_factors))

            ##################################################################################################################################
            # no in node and no out node linked to the factors in the cycle
            if len(factor_parentNode_childNode_outsideCycle) == 0:
                print("There are no in and out interaction factors in the cycle!")
                collapse_node = "collapse_node_" + "_".join(
                    list(
                        map(
                            lambda x: x.split("_")[1],
                            list(startAnchorNode_factors_nextAnchorNode.keys()),
                        )
                    )
                )
                factors_nodes.loc[:, collapse_node] = 0

                for anchor_node_i in startAnchorNode_factors_nextAnchorNode.keys():
                    all_drop_nodes.add(anchor_node_i)

                    parent_factor_outsideCycle = (
                        list(
                            filter(
                                lambda x: (x not in factors_in_cycle_set),
                                DG.predecessors(anchor_node_i),
                            )
                        )
                        + []
                    )
                    if parent_factor_outsideCycle:
                        for parent_factor_i in parent_factor_outsideCycle:
                            factors_nodes.loc[parent_factor_i, collapse_node] = -1

                    child_factor_outsideCycle = (
                        list(
                            filter(
                                lambda x: (x not in factors_in_cycle_set),
                                DG.successors(anchor_node_i),
                            )
                        )
                        + []
                    )
                    if child_factor_outsideCycle:
                        for child_factor_i in child_factor_outsideCycle:
                            factors_nodes.loc[child_factor_i, collapse_node] = 1

            else:
                #################################################################################################################################
                # at least one factor has in node and out node in the cycle
                start_anchor_node = find_start_anchor_node(
                    startAnchorNode_factors_nextAnchorNode
                )
                collapse_factors = []
                collapse_factor_id = 1

                cur_anchor_node = start_anchor_node
                while True:
                    next_anchor_node = startAnchorNode_factors_nextAnchorNode[
                        cur_anchor_node
                    ]["next_anchor_node"]
                    next_outInteraction_factors = (
                        startAnchorNode_factors_nextAnchorNode[cur_anchor_node][
                            "next_outInteraction_factors"
                        ]
                    )
                    # next_drop_factors = startAnchorNode_factors_nextAnchorNode[cur_anchor_node]['next_drop_factors']

                    # next anchor node is the start anchor node
                    if next_anchor_node == start_anchor_node:
                        # no interaction factors between the two anchor nodes
                        if len(next_outInteraction_factors) == 0:
                            break
                        else:
                            # there are out interaction factors between the two anchor nodes
                            collapse_factors.extend(next_outInteraction_factors)

                    else:
                        # no interaction factors between the two anchor nodes
                        if len(next_outInteraction_factors) == 0:
                            collapse_factor = "collapse_factor_" + str(
                                collapse_factor_id
                            ).zfill(2)
                            collapse_factor_id += 1
                            factors_nodes.loc[collapse_factor, :] = 0
                            factors_nodes.loc[collapse_factor, cur_anchor_node] = 1
                            factors_nodes.loc[collapse_factor, next_anchor_node] = -1

                        else:
                            # there are out node interaction factors between the two anchor nodes
                            collapse_factors.extend(next_outInteraction_factors)

                            collapse_factor = "collapse_factor_" + "_".join(
                                next_outInteraction_factors
                            )
                            factors_nodes.loc[collapse_factor, :] = 0
                            factors_nodes.loc[collapse_factor, cur_anchor_node] = 1
                            factors_nodes.loc[collapse_factor, next_anchor_node] = -1
                            for interaction_factor_i in next_outInteraction_factors:
                                all_drop_factors.add(interaction_factor_i)

                                parent_nodes = factor_parentNode_childNode_outsideCycle[
                                    interaction_factor_i
                                ]["outsideParentNode"]
                                child_nodes = factor_parentNode_childNode_outsideCycle[
                                    interaction_factor_i
                                ]["outsideChildNode"]
                                if parent_nodes:
                                    for parent_node_i in parent_nodes:
                                        # if parent_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[
                                            collapse_factor, parent_node_i
                                        ] = 1
                                        # visited_outside_nodes.add(parent_node_i)

                                if child_nodes:
                                    for child_node_i in child_nodes:
                                        # if child_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[
                                            collapse_factor, child_node_i
                                        ] = -1
                                        # visited_outside_nodes.add(child_node_i)

                            # collapse all the interactive factors into one factor and link all outside nodes to this collapse_factor
                            collapse_factor = "collapse_factor_" + "_".join(
                                collapse_factors
                            )

                            # the new collapse factor is already created then don't need to create it again
                            if collapse_factor in set(factors_nodes.index.values):
                                # collapse_factor="collapse_factor_"+'_'.join(collapse_factors)+'_'+str(collapse_factor_id).zfill(2)
                                # collapse_factor_id+=1
                                continue

                            factors_nodes.loc[collapse_factor, :] = 0
                            for interaction_factor_i in collapse_factors:
                                parent_nodes = factor_parentNode_childNode_outsideCycle[
                                    interaction_factor_i
                                ]["outsideParentNode"]
                                child_nodes = factor_parentNode_childNode_outsideCycle[
                                    interaction_factor_i
                                ]["outsideChildNode"]
                                if parent_nodes:
                                    for parent_node_i in parent_nodes:
                                        # if parent_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[
                                            collapse_factor, parent_node_i
                                        ] = 1
                                        # visited_outside_nodes.add(parent_node_i)

                                if child_nodes:
                                    for child_node_i in child_nodes:
                                        # if parent_node_i in visited_outside_nodes:
                                        #    continue
                                        factors_nodes.loc[
                                            collapse_factor, child_node_i
                                        ] = -1
                                        # visited_outside_nodes.add(child_node_i)

    # remove all drop factors and nodes
    factors_nodes.drop(all_drop_nodes, axis=1, inplace=True)
    factors_nodes.drop(all_drop_factors, axis=0, inplace=True)

    return factors_nodes


def build_bipartite_graph(factors_nodes):
    # generate graph
    BDG = nx.DiGraph()

    node_names = factors_nodes.columns.values
    factor_names = factors_nodes.index.values
    # add node to graph
    for node_name in node_names:
        # get parent factors
        idx = None
        idx = np.where(factors_nodes.loc[:, node_name] == -1)[0]
        parent_factors = list(np.take(factor_names, idx)) if len(idx) else []
        # get child factors
        idx = None
        idx = np.where(factors_nodes.loc[:, node_name] == 1)[0]
        child_factors = list(np.take(factor_names, idx)) if len(idx) else []
        BDG.add_node(
            node_name,
            bipartite=1,
            type="node",
            color="red",
            shape="o",
            n_parent=len(parent_factors),
            n_child=len(child_factors),
        )

        if len(parent_factors) == 0:
            dummy_factor = "dummy_parent_4_" + node_name
            BDG.add_node(
                dummy_factor,
                bipartite=0,
                color="blue",
                shape="s",
                n_parent=0,
                n_child=1,
            )
            BDG.add_edge(dummy_factor, node_name)
        if len(child_factors) == 0:
            dummy_factor = "dummy_child_4_" + node_name
            BDG.add_node(
                dummy_factor,
                bipartite=0,
                color="blue",
                shape="s",
                n_parent=1,
                n_child=0,
            )
            BDG.add_edge(node_name, dummy_factor)

    for factor_name in factor_names:
        # get parent nodes
        idx = None
        idx = np.where(factors_nodes.loc[factor_name, :] == 1)[0]
        parent_nodes = list(np.take(node_names, idx)) if len(idx) else []

        # get child nodes
        idx = None
        idx = np.where(factors_nodes.loc[factor_name, :] == -1)[0]
        child_nodes = list(np.take(node_names, idx)) if len(idx) else []
        # add factor to graph
        BDG.add_node(
            factor_name,
            bipartite=0,
            type="factor",
            color="blue",
            shape="s",
            n_parent=len(parent_nodes),
            n_child=len(child_nodes),
        )

        # add edge from parent node to factor
        for parent_node_i in parent_nodes:
            BDG.add_edge(parent_node_i, factor_name)

        for child_node_i in child_nodes:
            BDG.add_edge(factor_name, child_node_i)  # generate graph

    factors = [
        factor_i for factor_i in BDG.nodes() if BDG.nodes[factor_i]["bipartite"] == 0
    ]
    nodes = [node_i for node_i in BDG.nodes() if BDG.nodes[node_i]["bipartite"] == 1]

    return BDG, factors, nodes


def save_CycleCollapsed_factors_nodes(factors_nodes, BDG, cycles_in_graph, args):
    has_cycle = False if len(cycles_in_graph) == 0 else True

    if not has_cycle:
        print("\nNo Cycles in Graph!\n")
        return None, None

    print("\nRunning Cycle Collapsing...............................\n")

    factors_nodes_collapsedCycles = factors_nodes.copy()

    cycle_id = 1
    while len(cycles_in_graph) > 0:
        print(SEP_SIGN)
        print("\n There are cycles in the graph\n Processing the cycles in the graph\n")
        print("Factors and Nodes in cycle:\n{0}\n".format(cycles_in_graph))

        factors_nodes_collapsedCycles = collapse_cycles(
            factors_nodes_collapsedCycles.copy(), BDG, cycles_in_graph
        )
        factors_nodes_collapsedCycles = factors_nodes_collapsedCycles.astype(int)
        print(factors_nodes_collapsedCycles)
        print("\nCycle {0} collapsed!\n".format(cycle_id))
        cycle_id += 1

        BDG = None
        cycles_in_graph = None
        BDG = build_bipartite_graph(factors_nodes_collapsedCycles)
        cycles_in_graph = get_cycles(BDG)
    save_path = (
        args.output_dir
        + args.compouns_modules_file_name.split(".")[0]
        + "_collapsedCycles.csv"
    )
    factors_nodes_collapsedCycles.to_csv(save_path)

    print("\n New Cycles Collapsed factors_nodes file saved!!\n")
    return BDG, factors_nodes_collapsedCycles


def plot_genes_importance(genes_importance, args):
    # genes_importance := samples - genes_importance
    out_dir = args.output_dir + "figs/genes_importance/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for module_i in genes_importance.keys():
        importance_i = genes_importance[module_i]
        importance_i.sort(key=lambda x: x[1], reverse=True)
        # plot_genes_importance
        plt.bar(
            range(len(importance_i)),
            [importance for _, importance in importance_i],
            align="center",
        )
        plt.xticks(
            range(len(importance_i)), [gene for gene, _ in importance_i], rotation=90
        )
        plt.title(f"Genes Importance in {module_i}")
        plt.savefig(
            out_dir + module_i + "_gene_importance.png", bbox_inches="tight", dpi=200
        )
        plt.close()
    return True


def plot_loss_curve_for_scfea(imbalance_loss, cv_loss, args, fig_name="NNs"):
    # fig, ax = plt.subplots()
    x = list(range(len(imbalance_loss)))
    plt.figure(figsize=(6, 20))

    plt.subplot(2, 1, 1)
    plt.plot(x, imbalance_loss, marker="o", label="imbalance loss")
    plt.xlabel("iteration")
    plt.ylabel("imbalance loss")
    plt.title(fig_name + " :samples wise imbalance loss")
    plt.legend()
    # plt.show()
    # save_path=args.project_dir+args.res_dir+args.data_name+'_'+args.module_source+'_flux_scfea_bp_snn_std.png'
    # fig.savefig(save_path)

    plt.subplot(2, 1, 2)
    plt.plot(x, cv_loss, marker="o", label="cv loss")
    plt.xlabel("iteration")
    plt.ylabel("cv loss")
    plt.ylim(bottom=0)
    plt.title(fig_name + " :mean(coefficient of variation)")
    plt.legend()
    # plt.show()

    save_dir = args.output_dir + "figs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + fig_name + "_loss_curve.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    return True


def plot_loss_curve_for_mpo_regression(
    imbalance_list_mpo,
    cv_list_mpo,
    imbalance_list_regression,
    cv_list_regression,
    imbalance_scfea,
    cv_scfea,
    args,
    fig_name="NNs_MPO_regression",
):
    # fig, ax = plt.subplots()
    x = list(range(len(imbalance_list_mpo)))
    plt.figure(figsize=(6, 20))

    plt.subplot(2, 1, 1)
    plt.plot(x, imbalance_list_mpo, marker="o", label="imbalance loss target")
    plt.plot(
        x, imbalance_list_regression, marker="d", label="imbalance loss regression"
    )
    plt.axhline(imbalance_scfea, color="r", linestyle="-")
    plt.xlabel("iteration")
    plt.ylabel("imbalance loss")
    plt.ylim(bottom=0)
    plt.title(fig_name + " :samples-wise imbalance loss")
    plt.legend()
    # plt.show()
    # save_path=args.project_dir+args.res_dir+args.data_name+'_'+args.module_source+'_flux_scfea_bp_snn_std.png'
    # fig.savefig(save_path)

    plt.subplot(2, 1, 2)
    plt.plot(x, cv_list_mpo, marker="o", label="cv loss target")
    plt.plot(x, cv_list_regression, marker="d", label="cv loss regression")
    plt.axhline(cv_scfea, color="r", linestyle="-")
    plt.xlabel("iteration")
    plt.ylabel("cv loss")
    plt.ylim(bottom=0)
    plt.title(fig_name + " :sample-wise mean(coefficient of variation)")
    plt.legend()
    # plt.show()

    save_dir = args.output_dir + "figs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + fig_name + "_loss_curve.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    return True


def plot_mse_for_regression(mse_list_regression, args, fig_name="regression"):
    # fig, ax = plt.subplots()
    x = list(range(len(mse_list_regression)))
    plt.figure(figsize=(6, 20))

    plt.subplot(1, 1, 1)
    plt.plot(x, mse_list_regression, marker="o", label="mse loss regression")
    plt.xlabel("iteration")
    plt.ylabel("mse loss")
    plt.ylim(bottom=0)
    plt.title(fig_name + " :mean square error")
    plt.legend()
    # plt.show()

    save_dir = args.output_dir + "figs/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + fig_name + "_mse_curve.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    return True
