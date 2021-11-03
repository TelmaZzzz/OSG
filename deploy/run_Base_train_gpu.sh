#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

MODEL="Base_BART/LOT"
# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="fnlp/bart-large-chinese"
PRETRAIN="fnlp/bart-large-chinese"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
# TRAIN_PATH="$HOME/Datasets/LOT/data/train.jsonl"
# TRAIN_PATH="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train.jsonl"
TRAIN_PATH="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_7.jsonl"

python -m torch.distributed.launch --nproc_per_node 3 ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.00003 \
--batch_size=3 \
--epoch=30 \
--opt_step=3 \
# --model_load="$HOME/opt/tiger/polish/model/Base_BART/LOT/2021_10_27_23_25_epoch27.pkl" \
# > ../log/Base.log 2>&1 &