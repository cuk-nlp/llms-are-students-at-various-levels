#!/bin/bash

# LLASA without LLMDA
python llasa/report.py \
    --test_dataset dk_test_question.json \
    --llasa_result llasa_dk_cluster_4_llmda_False_iter_1.csv \
    --data_type dk \
    --llasa_type basic \
    --is_llmda False \
    --classification_setting_num 6