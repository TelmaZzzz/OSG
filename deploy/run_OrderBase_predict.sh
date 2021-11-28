#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
# MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_11_02_23_12_epoch44.pkl"      # encoder 0.8
# MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_11_03_01_24_epoch60.pkl"      # encoder 1.2
# MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_10_31_14_30_epoch40.pkl"      # encoder 1.0
# MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_11_02_18_10_epoch40.pkl"      # encoder 1.0 lr 6e-5
# MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_11_01_01_35_epoch40.pkl"      # encoder 1.0 permute 3
# MODEL="/users10/lyzhang/opt/tiger/polish/best/2021_11_04_04_57_epoch25.pkl"                 # keyword permute 6 37.43
MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_11_04_04_57_epoch38.pkl"      # keyword permute 6 37.26
# MODEL="/users10/lyzhang/opt/tiger/polish/model/OrderBase/2021_11_05_14_58_epoch42.pkl"      # encoder keyword permute 6 linear(0.8-0) 37.02
# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="fnlp/bart-large-chinese"
PRETRAIN="fnlp/bart-large-chinese"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
TRAIN_PATH="$HOME/Datasets/LOT/data/train.jsonl"

python -m torch.distributed.launch --nproc_per_node 2 ../src/OrderBase.py \
--predict \
--test_path="/users10/lyzhang/opt/tiger/polish/data/LOT/test.jsonl" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--model_load="$MODEL" \
--batch_size=4 \
--output="$HOME/opt/tiger/polish/output/OrderBase" \
--ans_list="$HOME/opt/tiger/polish/output/Base_ans.jsonl" \
> ../log/Base_predict.log 2>&1 &