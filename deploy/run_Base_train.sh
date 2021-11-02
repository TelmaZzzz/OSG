#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL="Base_BART"
# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="$HOME/model/LOT_large"
# TOKENIZER="fnlp/bart-large-chinese"
# PRETRAIN="fnlp/bart-large-chinese"
PRETRAIN="$HOME/model/LOT_large"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
TRAIN_PATH="$HOME/Datasets/LOT/data/train.jsonl"

# python -m torch.distributed.launch --nproc_per_node 2 ../src/Base.py \
python ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.00003 \
--batch_size=1 \
--epoch=2 \
--opt_step=2 \
> ../log/Base.log 2>&1 &