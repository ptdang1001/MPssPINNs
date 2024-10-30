# -*-coding:utf-8-*-

# build-in libraries
import sys
import argparse
import warnings

# Third-party libraries
import numpy as np

# import magic

# my libraries
from utils.data_interface import load_module
from utils.data_interface import load_geneExpression
from utils.data_interface import load_modulesGenes
from utils.data_interface import intersect_samples_genes
from utils.data_interface import pca_components_selection
from utils.data_interface import get_output_path
from utils.data_interface import get_metabolism
from utils.data_interface import get_imbalance_cv_loss
from utils.data_interface import plot_loss_curve_for_scfea
from utils.data_interface import plot_loss_curve_for_mpo_regression
from utils.data_interface import plot_mse_for_regression
from utils.data_interface import plot_genes_importance

from scFEA.src.scFEA import scFEA
from MPO_LoopOrAnchor.mpo import mpo
from Regression.regression import regression

# global variables
SEP_SIGN = "*" * 100


# @pysnooper.snoop()
def main(args):
    print(SEP_SIGN)
    print("Current Input parameters:\n{0}\n".format(args))
    print(SEP_SIGN)

    # set the seed
    # L.seed_everything(args.seed)
    # np.random.seed(args.seed)

    # load gene expression data
    # geneExpression is the gene expression data, cols:=samples/cells, rows:=genes, but the data will be transposed to rows:=samples/cells, cols:=genes automatically
    geneExpression = load_geneExpression(args)

    # load the modules(reactions) and the contained genes
    modules_genes = load_modulesGenes(args)

    # load the compounds and the reactions data, it is an adj matrix
    # compouns_modules is the adj matrix of the factor graph (reaction graph), rows:=compounds, columns:=reactions, entries are 0,1,-1
    compounds_modules, args = load_module(args)

    # remove non overlap genes
    # scaler = preprocessing.MinMaxScaler()
    # scaler = preprocessing.StandardScaler()
    scaler = None
    (
        geneExpression,
        modules_genes,
        compounds_modules,
        samples_mean,
        modules_normalizedGenes,
        no_genes_modules,
    ) = intersect_samples_genes(
        geneExpression, modules_genes, compounds_modules, scaler
    )
    if len(geneExpression) == 0:
        print("\n No Intersection of Genes between Data and (Modules)Reactions! \n")
        return False

    # imputation if too many missing values
    if args.do_imputation == True:
        magic_operator = magic.MAGIC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geneExpression = magic_operator.fit_transform(geneExpression)
        geneExpression = geneExpression.abs()
        print(
            f"gene expression sample after imputation:\n {geneExpression.iloc[np.random.choice(range(0, geneExpression.shape[0]), 10, replace=False), :]} \n"
        )

    # pca_components_selection, the default is False, which means no pca_components_selection
    if args.pca_components_selection == True:
        geneExpression_pca, modules_genes_pca = pca_components_selection(
            geneExpression, modules_genes, n_components=0.9
        )
        geneExpression = geneExpression_pca
        modules_genes = modules_genes_pca

    # prepare the input and output dir
    args.output_dir, _ = get_output_path(args)

    # save the gene expression data
    if args.save_normalized_geneExpression == True:
        n_genes, n_samples = geneExpression.shape
        geneExpression.T.to_csv(
            args.output_dir
            + "data_gene"
            + str(n_genes)
            + "_sample"
            + str(n_samples)
            + ".csv.gz",
            index=True,
            header=True,
            compression="gzip",
        )

    # the alg starts here
    # all steps: scfea->mpo<->regression(training,testing,predicting)

    flux_scfea_final = None
    metabolites_scfea_final = None
    flux_scfea_mpo_final = None
    metabolites_scfea_mpo_final = None
    flux_mpo_final = None
    metabolites_mpo_final = None
    flux_regression_final = None
    metabolites_regression_final = None
    modules_genes_importance_regression_final = None

    #########################################################################################################################
    # 1st step to generate the initial values by scFEA
    samples_modules_scfea, loss_cv_imbalance_scfea_epoches = scFEA(
        geneExpression.copy(),
        modules_genes.copy(),
        compounds_modules.copy(),
        samples_mean,
        args,
    )
    if len(no_genes_modules) > 0:
        samples_modules_scfea[no_genes_modules] = 0.0
    samples_modules_scfea = samples_modules_scfea[compounds_modules.columns]
    samples_modules_scfea = samples_modules_scfea.loc[geneExpression.index, :]
    flux_scfea_final = samples_modules_scfea.copy()
    imbalance_cv_loss_scfea = get_imbalance_cv_loss(
        compounds_modules.copy(), samples_modules_scfea.values.copy()
    )
    print(f"imbalance_cv_loss_scfea: {imbalance_cv_loss_scfea} \n")
    metabolites_scfea = get_metabolism(
        compounds_modules.copy(), samples_modules_scfea.copy()
    )
    metabolites_scfea_final = metabolites_scfea.copy()
    """
    print(
       f"scfea flux:\n {samples_modules_scfea.iloc[np.random.choice(range(0, geneExpression.shape[0]), 10, replace=False), :]} \n")
    print(
       f"metabolites_scfea:\n {metabolites_scfea.iloc[np.random.choice(range(0, metabolites_scfea.shape[0]), 10, replace=False), :]} \n")
    """
    flux_scfea_final.to_csv(f"{args.output_dir}flux_scfea.csv", index=True, header=True)
    metabolites_scfea_final.to_csv(
        f"{args.output_dir}metabolites_scfea.csv", index=True, header=True
    )
    # plot the scfea loss values
    plot_loss_curve_for_scfea(
        loss_cv_imbalance_scfea_epoches["loss_imbalance"],
        loss_cv_imbalance_scfea_epoches["loss_cv"],
        args,
    )

    # add pertabation to the scfea result
    min_value = samples_modules_scfea.min().min()
    max_value = samples_modules_scfea.max().max()
    sd_pertabation = max(0.1 * (max_value - min_value), 0.1 * min_value)
    # sd_pertabation = 10
    samples_modules_scfea = samples_modules_scfea + np.random.normal(
        0, sd_pertabation, samples_modules_scfea.shape
    )
    samples_modules_scfea = samples_modules_scfea.abs()

    print("\n scFEA Done! \n")

    #########################################################################################################################
    samples_modules_input_for_mpo = samples_modules_scfea.copy()

    # threshold to choose regression's result
    minImbalanceMaxCv_flux_regression = float("inf")
    # threshold to update mpo's result as label
    minImbalanceMaxCv_flux_mpo = float("inf")
    min_imbalance_loss_mpo = float("inf")
    max_cv_loss_mpo = float("-inf")
    min_imbalance_loss_regression = float("inf")
    max_cv_loss_regression = float("-inf")
    # threshold to choose stop
    minImbalanceMaxCv_flux_stop = float("inf")
    samples_modules_target_for_regression = None

    all_imbalance_loss_mpo = []
    all_imbalance_loss_regression = []
    all_cv_loss_mpo = []
    all_cv_loss_regression = []
    all_mse_regression = []

    cur_step = 0
    while cur_step < args.n_epoch_all:
        print(SEP_SIGN)
        print("\n The {0}th epoch_all! \n".format(cur_step + 1))
        print(SEP_SIGN)

        #########################################################################################################################
        # 2nd step to improve the initial values by Massage Passing - based flux balancing
        print("\n Massage Passing-based flux balancing! \n")
        main_branch = []
        samples_modules_mpo = mpo(
            compounds_modules.copy(),
            samples_modules_input_for_mpo.copy(),
            main_branch,
            samples_mean,
            args,
        )
        samples_modules_mpo = samples_modules_mpo[compounds_modules.columns]
        samples_modules_mpo = samples_modules_mpo.loc[geneExpression.index, :]
        imbalance_cv_loss_mpo = get_imbalance_cv_loss(
            compounds_modules.copy(), samples_modules_mpo.values.copy()
        )
        """
        print(f"imbalance_cv_loss_mpo: {imbalance_cv_loss_mpo} \n")
        metabolites_mpo = get_metabolism(compounds_modules, samples_modules_mpo)
        print(
           f"mpo flux:\n {samples_modules_mpo.iloc[np.random.choice(range(0, samples_modules_mpo.shape[0]), 10, replace=False), :]} \n")
        print(
           f"metabolites_mpo:\n {metabolites_mpo.iloc[np.random.choice(range(0, metabolites_mpo.shape[0]), 10, replace=False), :]} \n")
        """
        if cur_step == 0 and (
            imbalance_cv_loss_mpo["mean_imbalance"] - imbalance_cv_loss_mpo["mean_cv"]
        ) > (
            imbalance_cv_loss_scfea["mean_imbalance"]
            - imbalance_cv_loss_scfea["mean_cv"]
        ):
            samples_modules_mpo = (samples_modules_mpo + samples_modules_scfea) / 2.0
        imbalance_cv_loss_mpo = get_imbalance_cv_loss(
            compounds_modules.copy(), samples_modules_mpo.values.copy()
        )
        print(f"imbalance_cv_loss_mpo: {imbalance_cv_loss_mpo} \n")
        metabolites_mpo = get_metabolism(
            compounds_modules.copy(), samples_modules_mpo.copy()
        )

        if cur_step == 0:
            flux_scfea_mpo_final = samples_modules_mpo.copy()

        print(f"Current Message Passing Step Done!\n")
        # sys.exit(1)

        #########################################################################################################################

        # 3rd step to improve the flux values by regression regression

        # update the target for regression
        if cur_step == 0:
            samples_modules_target_for_regression = samples_modules_mpo.copy()
            flux_scfea_mpo_final = samples_modules_mpo.copy()
            metabolites_scfea_mpo_final = metabolites_mpo.copy()
            flux_mpo_final = samples_modules_mpo.copy()
            metabolites_mpo_final = metabolites_mpo.copy()
            # minImbalanceMaxCv_flux_mpo = imbalance_cv_loss_mpo['mean_imbalance'] - imbalance_cv_loss_mpo['mean_cv']
            min_imbalance_loss_mpo = imbalance_cv_loss_mpo["mean_imbalance"]
            max_cv_loss_mpo = imbalance_cv_loss_mpo["mean_cv"]

            flux_scfea_mpo_final.to_csv(
                f"{args.output_dir}flux_scfea_mpo.csv", index=True, header=True
            )
            metabolites_scfea_mpo_final.to_csv(
                f"{args.output_dir}metabolites_scfea_mpo.csv", index=True, header=True
            )

        else:
            if (
                imbalance_cv_loss_mpo["mean_cv"] >= max_cv_loss_mpo
                and imbalance_cv_loss_mpo["mean_imbalance"] <= min_imbalance_loss_mpo
            ):
                if np.round(imbalance_cv_loss_mpo["mean_cv"], 2) == np.round(
                    max_cv_loss_mpo, 2
                ) and np.round(imbalance_cv_loss_mpo["mean_imbalance"], 2) == np.round(
                    min_imbalance_loss_mpo, 2
                ):
                    break
                samples_modules_target_for_regression = samples_modules_mpo.copy()
                flux_mpo_final = samples_modules_mpo.copy()
                metabolites_mpo_final = metabolites_mpo.copy()
                # minImbalanceMaxCv_flux_mpo = imbalance_cv_loss_mpo['mean_cv']
                min_imbalance_loss_mpo = imbalance_cv_loss_mpo["mean_imbalance"]
                max_cv_loss_mpo = imbalance_cv_loss_mpo["mean_cv"]

        imbalance_cv_loss_regression_target = get_imbalance_cv_loss(
            compounds_modules.copy(),
            samples_modules_target_for_regression.values.copy(),
        )
        all_imbalance_loss_mpo.append(
            imbalance_cv_loss_regression_target["mean_imbalance"]
        )
        all_cv_loss_mpo.append(imbalance_cv_loss_regression_target["mean_cv"])

        print(
            f"\nimbalance loss regression target:{min_imbalance_loss_mpo}, cv loss regression target:{max_cv_loss_mpo} \n"
        )

        (
            samples_modules_regression,
            mse_steps_regression,
            modules_genes_importance_regression,
        ) = regression(
            samples_modules_target_for_regression,
            samples_mean,
            modules_normalizedGenes,
            compounds_modules.copy(),
            cur_step,
            args,
        )
        samples_modules_regression = samples_modules_regression[
            compounds_modules.columns
        ]
        samples_modules_regression = samples_modules_regression.loc[
            geneExpression.index, :
        ]
        # metabolites_regression = get_metabolism(compounds_modules, samples_modules_regression)
        imbalance_cv_loss_regression = get_imbalance_cv_loss(
            compounds_modules.copy(), samples_modules_regression.values.copy()
        )
        print(
            f"imbalance_loss_regression: {imbalance_cv_loss_regression['mean_imbalance']}, cv_loss_regression: {imbalance_cv_loss_regression['mean_cv']} \n"
        )
        all_imbalance_loss_regression.append(
            imbalance_cv_loss_regression["mean_imbalance"]
        )
        all_cv_loss_regression.append(imbalance_cv_loss_regression["mean_cv"])
        all_mse_regression.append(np.mean(mse_steps_regression))
        # update the final flux for regression
        if (
            imbalance_cv_loss_regression["mean_imbalance"]
            - imbalance_cv_loss_regression["mean_cv"]
        ) <= minImbalanceMaxCv_flux_regression:
            flux_regression_final = samples_modules_regression.copy()
            min_imbalance_loss_regression = imbalance_cv_loss_regression[
                "mean_imbalance"
            ]
            max_cv_loss_regression = imbalance_cv_loss_regression["mean_cv"]
            metabolites_regression_final = get_metabolism(
                compounds_modules.copy(), flux_regression_final.copy()
            )
            modules_genes_importance_regression_final = (
                modules_genes_importance_regression.copy()
            )
            minImbalanceMaxCv_flux_regression = (
                imbalance_cv_loss_regression["mean_imbalance"]
                - imbalance_cv_loss_regression["mean_cv"]
            )
            minImbalanceMaxCv_flux_stop = minImbalanceMaxCv_flux_regression

        """
        print(
           f"regression flux:\n {samples_modules_regression.iloc[np.random.choice(range(0, samples_modules_regression.shape[0]), 10, replace=False), :]} \n")
        print(
           f"metabolites_regression:\n {metabolites_regression.iloc[np.random.choice(range(0, metabolites_regression.shape[0]), 10, replace=False), :]} \n")
        """
        # update the step number
        cur_step += 1

        print("\n Current Regression Done! \n")

        #########################################################################################################################
        # update the input for the MPO for the next iteration
        samples_modules_input_for_mpo = samples_modules_regression.copy()

        # plot the loss values

        # plot the mpo_regression loss values
        plot_loss_curve_for_mpo_regression(
            all_imbalance_loss_mpo,
            all_cv_loss_mpo,
            all_imbalance_loss_regression,
            all_cv_loss_regression,
            imbalance_cv_loss_scfea["mean_imbalance"],
            imbalance_cv_loss_scfea["mean_cv"],
            args,
        )
        # plot the mse values
        plot_mse_for_regression(all_mse_regression, args)  #
        # save final flux res

        flux_mpo_final.to_csv(f"{args.output_dir}flux_mpo.csv", index=True, header=True)
        metabolites_mpo_final.to_csv(
            f"{args.output_dir}metabolites_mpo.csv", index=True, header=True
        )
        if (
            min_imbalance_loss_regression > imbalance_cv_loss_scfea["mean_imbalance"]
            and max_cv_loss_regression < imbalance_cv_loss_scfea["mean_cv"]
        ):
            flux_regression_final = (flux_regression_final + flux_mpo_final) / 2.0
            metabolites_regression_final = (
                metabolites_regression_final + metabolites_mpo_final
            ) / 2.0
        flux_regression_final.to_csv(
            f"{args.output_dir}flux_regression.csv", index=True, header=True
        )
        metabolites_regression_final.to_csv(
            f"{args.output_dir}metabolites_regression.csv", index=True, header=True
        )
        plot_genes_importance(modules_genes_importance_regression_final, args)

    # plot the loss values
    # plot the mpo_regression loss values
    plot_loss_curve_for_mpo_regression(
        all_imbalance_loss_mpo,
        all_cv_loss_mpo,
        all_imbalance_loss_regression,
        all_cv_loss_regression,
        imbalance_cv_loss_scfea["mean_imbalance"],
        imbalance_cv_loss_scfea["mean_cv"],
        args,
    )
    # plot the mse values
    plot_mse_for_regression(all_mse_regression, args)  #
    flux_mpo_final.to_csv(f"{args.output_dir}flux_mpo.csv", index=True, header=True)
    metabolites_mpo_final.to_csv(
        f"{args.output_dir}metabolites_mpo.csv", index=True, header=True
    )
    if (
        min_imbalance_loss_regression > imbalance_cv_loss_scfea["mean_imbalance"]
        and max_cv_loss_regression < imbalance_cv_loss_scfea["mean_cv"]
    ):
        flux_regression_final = (flux_regression_final + flux_mpo_final) / 2.0
        metabolites_regression_final = (
            metabolites_regression_final + metabolites_mpo_final
        ) / 2.0
    flux_regression_final.to_csv(
        f"{args.output_dir}flux_regression.csv", index=True, header=True
    )
    metabolites_regression_final.to_csv(
        f"{args.output_dir}metabolites_regression.csv", index=True, header=True
    )
    plot_genes_importance(modules_genes_importance_regression_final, args)

    print(SEP_SIGN)
    print("Current Input parameters:\n{0}\n".format(args))
    print(SEP_SIGN)

    print("\n Done! \n")
    return True


def parse_arguments(parser):
    # global parameters
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--input_dir", type=str, default="./inputs/", help="The inputs directory."
    )
    parser.add_argument(
        "--network_dir",
        type=str,
        default="./inputs/module_info/",
        help="The project directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/",
        help="The outputs directory, you can find all outputs in this directory.",
    )
    parser.add_argument(
        "--geneExpression_file_name",
        type=str,
        default="NA",
        help="The scRNA-seq file name.",
    )
    parser.add_argument(
        "--compounds_modules_file_name",
        type=str,
        default="NA",
        help="The table describes relationship between compounds and modules. Each row is an intermediate metabolite and each column is metabolic module. For human model, please use cmMat_171.csv which is default. All candidate stoichiometry matrices are provided in /data/ folder.",
    )
    parser.add_argument(
        "--modules_genes_file_name",
        type=str,
        default="NA",
        help="The json file contains genes for each module. We provide human and mouse two models in scFEA.",
    )
    parser.add_argument(
        "--n_epoch_all",
        type=int,
        default=10,
        help="The user defined early stop Epoch(the whole framework)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="The user defined early stop imbalance loss.",
    )
    parser.add_argument(
        "--pca_components_selection",
        type=bool,
        default=False,
        help="Apply PCA to reduce the dimension of features. False or True",
    )
    parser.add_argument(
        "--do_imputation",
        type=bool,
        default=False,
        help="Imputation on the input gene expression matrix. False or True",
    )
    parser.add_argument("--experiment_name", type=str, default="flux")
    parser.add_argument("--network_name", type=str, default="NA")
    parser.add_argument("--save_normalized_geneExpression", type=bool, default=False)

    # parameters for scFEA
    parser.add_argument(
        "--n_epoch_scfea",
        type=int,
        default=100,
        help="User defined Epoch for scFEA training.",
    )
    parser.add_argument(
        "--batch_size_scfea", type=int, default=1000000, help="Batch size, scfea."
    )

    # parameters for bp_balance
    parser.add_argument(
        "--n_epoch_mpo",
        type=int,
        default=100,
        help="User defined Epoch for Message Passing Optimizer.",
    )
    parser.add_argument(
        "--delta", type=float, default=0.001, help="delta for the stopping criterion"
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.4, help="beta_1 for the update step"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.5, help="beta_2 for main branch"
    )

    # parameters for regression

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # mp.set_start_method('spawn',force=True)
    parser = argparse.ArgumentParser(
        description="MPSSFE: A Massage Passing - Based self supervised Model to Estimate Cell-Wise Metabolic Using Single Cell RNA-seq Data."
    )

    # global args
    args = parse_arguments(parser)

    main(args)
