#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="Base_BART"
# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="fnlp/bart-large-chinese"
PRETRAIN="fnlp/bart-large-chinese"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
# TRAIN_PATH="$HOME/Datasets/LOT/data/train.jsonl"
TRAIN_PATH="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train_permute_5.jsonl"

python -m torch.distributed.launch --nproc_per_node 2 ../src/Base.py \
--predict \
--test_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--model_load="$HOME/opt/tiger/polish/best/2021_10_31_11_28_epoch24.pkl" \
--batch_size=3 \
--output="$HOME/opt/tiger/polish/output/Base" \
--ans_list="$HOME/opt/tiger/polish/output/Base_ans.jsonl" \
# > ../log/Base_predict.log 2>&1 &