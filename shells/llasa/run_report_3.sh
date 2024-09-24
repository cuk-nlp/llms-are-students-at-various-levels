#!/bin/bash

# Zero-shot LLASA
python llasa/report.py \
    --test_dataset dk_test_question.json \
    --llasa_result zeroshot_llasa_dk_low_150_middle_60_high_30.csv \
    --data_type dk \
    --llasa_type zeroshot \
    --classification_setting_num 6