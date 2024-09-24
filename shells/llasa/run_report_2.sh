#!/bin/bash

# LLASA with LLMDA
python llasa/report.py \
    --test_dataset dk_test_question.json \
    --llasa_result llasa_dk_cluster_4_llmda_True_iter_3.csv \
    --data_type dk \
    --llasa_type basic \
    --is_llmda True \
    --classification_setting_num 6