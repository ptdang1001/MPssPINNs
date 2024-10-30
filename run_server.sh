#!/bin/zsh

#SBATCH -J MPssPINNs
#SBATCH -p general
#SBATCH -A r00077
#SBATCH -o /N/slate/pdang/myProjectsDataRes/jobOutputs/%j.out
#SBATCH -e /N/slate/pdang/myProjectsDataRes/jobOutputs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pdang@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=128G

#module load

python MPssPINNs/src/main.py \
    --input_dir /N/slate/pdang/myProjectsDataRes/20211201FLUXOptimization/data/ \
    --output_dir /N/slate/pdang/myProjectsDataRes/20211201FLUXOptimization/result/MPSSFE_results/ \
    --geneExpression_file_name xbUTSW.csv \
    --compounds_modules_file_name module_info/Lipid_V9_cmMat.csv \
    --modules_genes_file_name module_info/Lipid_V9_modules_genes.json \
    --experiment_name flux \
    --network_name Lipid_V9 \
    --n_epoch_all 1 \
    --n_epoch_scfea 5 \
    --n_epoch_mpo 5
