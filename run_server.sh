#!/bin/zsh

# You may need to update some parameters depends on your server
#SBATCH -J MPssPINNs
#SBATCH -p xxxxx
#SBATCH -A xxxxxx
#SBATCH -o %j_log.out
#SBATCH -e %j_log.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxxxx@xxx.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=128G

#module load

python ./src/main.py \
    --input_dir ./inputs/Data/ \
    --network_dir ./inputs/network_info/ \
    --output_dir ./outputs/ \
    --geneExpression_file_name GSE72056_gene569_cell4486.csv.gz \
    --compounds_modules_file_name M171_V3_connected_cmMat.csv \
    --modules_genes_file_name M171_V3_connected_reactions_genes.json \
    --experiment_name Flux \
    --n_epoch_all 150 \
    --n_epoch_scfea 100 \
    --n_epoch_mpo 50
