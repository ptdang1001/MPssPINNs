# -*- coding: utf-8 -*-
"""

@author: wnchang@iu.edu
"""

# system lib
import time

# import warnings
# import sys

# tools
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd

# import magic
from tqdm import tqdm

# import lightning as L

# import pysnooper

# scFEA lib
from scFEA.src.ClassFlux import FLUX  # Flux class network
from scFEA.src.util import pearsonr
from scFEA.src.DatasetFlux import MyDataset

# hyper parameters
LEARN_RATE = 0.008
# EPOCH = 100
LAMB_BA = 1
LAMB_NG = 1
LAMB_CELL = 1
LAMB_MOD = 1e-2


def get_loss(compounds_modules, flux):
    flux = abs(flux)
    coefficient_of_variation = flux.std(axis=0) / flux.mean(axis=0)
    coefficient_of_variation = np.nan_to_num(coefficient_of_variation)

    # normalize the data by row
    flux = np.divide(flux, np.linalg.norm(flux, axis=1).reshape(-1, 1)) * 100.0
    imbalanceLoss_values = []
    for belief_i in flux:
        tmp1 = belief_i * compounds_modules
        tmp2 = np.sum(tmp1, axis=1)
        # tmp3 = tmp2 ** 2
        tmp3 = np.abs(tmp2)
        tmp4 = np.sum(tmp3)
        tmp5 = np.round(tmp4, 3)
        imbalanceLoss_values.append(tmp5)

    return {
        "mean_cv": np.mean(coefficient_of_variation),
        "mean_imbalanceLoss": np.mean(imbalanceLoss_values),
    }


def myLoss(
    m, c, lamb1=0.2, lamb2=0.2, lamb3=0.2, lamb4=0.2, geneScale=None, moduleScale=None
):
    # m: flxu, row:=samples, col:=reactions
    # c: balance, row:=samples, col:=compounds

    # balance constrain
    # total1 = torch.pow(c, 2)
    total1 = torch.abs(c)
    total1 = torch.sum(total1, dim=1)

    # non-negative constrain
    error = torch.abs(m) - m
    total2 = torch.sum(error, dim=1)

    # sample-wise variation constrain
    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    # diff = torch.abs(torch.sum(m, dim=1) - geneScale)
    # total3 = torch.pow(diff, 0.5)
    if sum(diff > 0) == m.shape[0]:  # solve Nan after several iteraions
        total3 = torch.pow(diff, 0.5)
    else:
        total3 = diff

    # module-wise variation constrain
    if lamb4 > 0:
        corr = torch.FloatTensor(np.ones(m.shape[0]))
        for i in range(m.shape[0]):
            corr[i] = pearsonr(m[i, :], moduleScale[i, :])
        corr = torch.abs(corr)
        penal_m_var = torch.FloatTensor(np.ones(m.shape[0])) - corr
        total4 = penal_m_var
    else:
        total4 = torch.FloatTensor(np.zeros(m.shape[0]))

    # loss
    loss1 = torch.sum(lamb1 * total1)
    if torch.isnan(loss1):
        loss1 = torch.FloatTensor(np.zeros(1))
    loss2 = torch.sum(lamb2 * total2)
    if torch.isnan(loss2):
        loss2 = torch.FloatTensor(np.zeros(1))
    loss3 = torch.sum(lamb3 * total3)
    if torch.isnan(loss3):
        loss3 = torch.FloatTensor(np.zeros(1))
    loss4 = torch.sum(lamb4 * total4)
    if torch.isnan(loss4):
        loss4 = torch.FloatTensor(np.zeros(1))
    loss = (loss1 + loss2 + loss3 + loss4) / 4.0
    return loss, loss1, loss2, loss3, loss4


def read_json2dict(file_path, file_name):
    import json

    # Opening JSON file
    f = open(file_path + file_name)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    return data


def check_flux_col_variation(flux, sampls_mean):
    flux = flux.multiply(sampls_mean, axis=0)
    return flux


# @pysnooper.snoop()
def scFEA(geneExpr, moduleGene, cmMat, sampls_mean, args):
    torch.manual_seed(args.seed)

    # set arguments
    BATCH_SIZE = min(args.batch_size_scfea, geneExpr.shape[0])
    EPOCH = args.n_epoch_scfea
    if EPOCH <= 0:
        raise NameError("EPOCH must greater than 1!")

    # choose cpu or gpu automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    print("scFEA, Starting load data...")

    # normalize the data by column min-max
    # by column
    geneExpr = (geneExpr - geneExpr.min()) / (geneExpr.max() - geneExpr.min())

    # normalize the data by log2
    if geneExpr.max().max() > 50:
        geneExpr = (geneExpr + 1).apply(np.log2)

    geneExprSum = geneExpr.sum(axis=1)
    stand = geneExprSum.mean()
    geneExprScale = geneExprSum / stand
    # print("geneExprScale: ", geneExprScale)
    geneExprScale = torch.FloatTensor(geneExprScale.values).to(device)

    modules = list(moduleGene.keys())
    # modules = list(set(modules).intersection(set(cmMat.columns)))
    # modules.sort()
    # find existing gene
    module_gene_all = []
    for module_i in modules:
        genes = moduleGene[module_i]
        module_gene_all.extend(genes)
    module_gene_all = set(module_gene_all)
    moduleLen = np.array([len(moduleGene[module_i]) for module_i in modules])
    data_gene_all = set(geneExpr.columns)
    gene_overlap = list(data_gene_all.intersection(module_gene_all))  # fix
    gene_overlap.sort()

    cmMat = cmMat[modules]
    cmMat = cmMat.values
    cmMat = torch.FloatTensor(cmMat).to(device)

    print("Load data done.")

    print("Starting process data...")
    # emptyNode = []
    # extract overlap gene
    geneExpr = geneExpr[gene_overlap]
    # print(f"{geneExpr=}")
    # print("geneExpr: ", geneExpr.shape)
    gene_names = geneExpr.columns
    cell_names = geneExpr.index.astype(str)
    # n_modules = moduleGene.shape[0]
    n_modules = len(moduleGene.keys())
    n_genes = len(gene_names)
    n_cells = len(cell_names)
    n_comps = cmMat.shape[0]
    geneExprDf = pd.DataFrame(columns=["Module_Gene"] + list(cell_names))
    # for i in range(n_modules):
    for i, module_i in enumerate(modules):
        # genes = moduleGene.iloc[i, :].values.astype(str)
        genes = moduleGene[module_i]
        genes = [g for g in genes if g != "nan"]
        if not genes:
            # emptyNode.append(i)
            continue
        temp = geneExpr.copy()
        temp.loc[:, [g for g in gene_names if g not in genes]] = 0
        temp = temp.T
        temp["Module_Gene"] = ["%02d_%s" % (i, g) for g in gene_names]
        # geneExprDf = geneExprDf._append(temp, ignore_index=True, sort=False) # pandas version update, append -> concat
        geneExprDf = pd.concat([geneExprDf, temp], ignore_index=True, sort=False)
    geneExprDf.index = geneExprDf["Module_Gene"]
    geneExprDf.drop("Module_Gene", axis="columns", inplace=True)
    X = geneExprDf.values.T
    # print(f"{X=}")
    # print("X shape: ", X.shape)
    # sys.exit(1)
    X = torch.FloatTensor(X).to(device)

    # prepare data for constraint of module variation based on gene
    df = geneExprDf
    df.index = [i.split("_")[0] for i in df.index]
    df.index = df.index.astype(
        int
    )  # mush change type to ensure correct order, T column name order change!
    # module_scale = df.groupby(df.index).sum(axis=1).T   # pandas version update
    module_scale = df.groupby(df.index).sum().T
    module_scale = torch.FloatTensor(module_scale.values / moduleLen)
    # print("module_scale: ", module_scale)
    print("Process data done.")
    # sys.exit(1)

    # =============================================================================
    # NN
    net = FLUX(X, n_modules, f_in=n_genes, f_out=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARN_RATE, weight_decay=1e-5)

    # Dataloader
    dataSet = MyDataset(X, geneExprScale, module_scale)
    dataloader_params_train = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=dataSet, **dataloader_params_train
    )

    # =============================================================================
    print("Starting train neural network...")
    start = time.time()
    #   training
    loss_v = []
    loss_v1 = []
    loss_v2 = []
    loss_v3 = []
    loss_v4 = []
    loss_imbalance = []
    loss_cv = []
    net.train()
    min_imbalance_loss = float("inf")
    max_cv = float("-inf")
    best_model_state_dict = None
    save_model_epoch = 0
    for epoch in tqdm(range(EPOCH)):
        loss, loss1, loss2, loss3, loss4 = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (X, X_scale, m_scale) in enumerate(train_loader):
            X_batch = Variable(X.float().to(device))
            X_scale_batch = Variable(X_scale.float().to(device))
            m_scale_batch = Variable(m_scale.float().to(device))

            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes, n_comps, cmMat)
            loss_cv_imbalance = get_loss(
                cmMat.cpu().numpy().copy(), out_m_batch.detach().cpu().numpy().copy()
            )
            loss_imbalance.append(loss_cv_imbalance["mean_imbalanceLoss"])
            loss_cv.append(loss_cv_imbalance["mean_cv"])
            if (
                loss_cv_imbalance["mean_imbalanceLoss"] <= min_imbalance_loss
                and loss_cv_imbalance["mean_cv"] >= max_cv
            ):
                min_imbalance_loss = loss_cv_imbalance["mean_imbalanceLoss"]
                max_cv = loss_cv_imbalance["mean_cv"]
                best_model_state_dict = net.state_dict().copy()
                save_model_epoch = epoch
            loss_batch, loss1_batch, loss2_batch, loss3_batch, loss4_batch = myLoss(
                out_m_batch,
                out_c_batch,
                lamb1=LAMB_BA,
                lamb2=LAMB_NG,
                lamb3=LAMB_CELL,
                lamb4=LAMB_MOD,
                geneScale=X_scale_batch,
                moduleScale=m_scale_batch,
            )

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            loss += loss_batch.cpu().data.numpy()
            loss1 += loss1_batch.cpu().data.numpy()
            loss2 += loss2_batch.cpu().data.numpy()
            loss3 += loss3_batch.cpu().data.numpy()
            loss4 += loss4_batch.cpu().data.numpy()

        # print('epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f, loss4: %.8f, loss: %.8f' % (epoch+1, loss1, loss2, loss3, loss4, loss))
        # file_loss.write('epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f, loss4: %.8f, loss: %.8f. \n' % (
        # epoch + 1, loss1, loss2, loss3, loss4, loss))

        loss_v.append(loss)
        loss_v1.append(loss1)
        loss_v2.append(loss2)
        loss_v3.append(loss3)
        loss_v4.append(loss4)

    loss = {"loss_imbalance": loss_imbalance, "loss_cv": loss_cv}

    # =============================================================================
    end = time.time()
    print("Training time: ", end - start)

    #    Dataloader
    dataloader_params = {
        "batch_size": n_cells,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }

    dataSet = MyDataset(X, geneExprScale, module_scale)
    test_loader = torch.utils.data.DataLoader(dataset=dataSet, **dataloader_params)

    # testing
    fluxStatuTest = np.zeros((n_cells, n_modules), dtype="f")  # float32
    # balanceStatus = np.zeros((n_cells, n_comps), dtype='f')
    if best_model_state_dict is not None:
        print(
            f"load the best model at epoch {save_model_epoch + 1} with min imbalance loss {min_imbalance_loss} and max cv {max_cv}"
        )
        net.load_state_dict(best_model_state_dict)
    net.eval()
    with torch.no_grad():
        for epoch in range(1):
            for i, (X, X_scale, _) in enumerate(test_loader):
                X_batch = Variable(X.float().to(device))
                # out_m_batch := flux
                # out_c_batch := compounds balance
                out_m_batch, out_c_batch = net(
                    X_batch, n_modules, n_genes, n_comps, cmMat
                )
                # print(f"out_m_batch shape: {out_m_batch.shape}")
                # print(f"out_c_batch shape: {out_c_batch.shape}")
                # save data
                fluxStatuTest = out_m_batch.detach().cpu().numpy()
                # balanceStatus = out_c_batch.detach().cpu().numpy()

    # save to file
    flux = pd.DataFrame(fluxStatuTest)
    # setF.columns = moduleGene.index
    flux.columns = modules
    flux.index = geneExpr.index.tolist()
    flux.fillna(0, inplace=True)
    flux = flux.abs()
    # if flux.std(axis=0).mean() < 0.1:
    if True:
        flux = flux.div(flux.sum(axis=1), axis=0) * (10.0 ** (len(str(len(modules)))))
        flux = check_flux_col_variation(flux, sampls_mean)
    return flux, loss
