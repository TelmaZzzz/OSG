#!/bin/sh
MAX_LENGTH=500
LABEL="fiction"
# CROUPS="bookcroups"
CROUPS="rake"
export PYTHONPATH="$HOME/opt/tiger/polish"
# python ../data_prepare/data_prepare.py \
# --dir_path="$HOME/Datasets/bookcroups/books1/epubtxt" \
# --cropus_type="$CROUPS" \
# --save_path="$HOME/opt/tiger/polish/data/bookcroups_$MAX_LENGTH.txt" \
# --max_length=$MAX_LENGTH \
# --prepare_rawdata \
# > ../log/data_prepare.log 2>&1 &
python ../data_prepare/data_prepare.py \
--rawdata_path="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2.jsonl" \
--cropus_type="$CROUPS" \
--save_path="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl" \
--max_length=$MAX_LENGTH \
--prepare_rawdata \
> ../log/data_prepare_2.log 2>&1 &