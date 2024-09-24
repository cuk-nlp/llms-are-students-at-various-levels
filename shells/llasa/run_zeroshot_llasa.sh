#!/bin/bash

python llasa/zeroshot_llasa.py \
    --data_type dk \
    --model_answer_log model_answer_log.csv \
    --train_dataset dk_train_question.json \
    --test_dataset dk_test_question.json \
    --low_level_llms 150 \
    --middle_level_llms 60 \
    --high_level_llms 30 \
    --load_history False