#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

python ../src/Ordering.py \
--build_data \
--train_path="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_5.jsonl" \
--valid_path="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_7.jsonl" \
--test_path="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_9.jsonl" \
--train_save="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_5_order.jsonl" \
--valid_save="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_7_order.jsonl" \
--test_save="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_9_order.jsonl" \
> ../log/Ordering.log 2>&1 &