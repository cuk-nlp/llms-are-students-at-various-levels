#!/bin/bash

python llasa/basic_llasa.py \
    --data_type dk \
    --model_answer_log model_answer_log.csv \
    --train_dataset dk_train_question.json \
    --train_transaction train_transaction.csv \
    --test_dataset dk_test_question.json \
    --test_transaction test_transaction.csv \
    --cluster_size 4 \
    --max_drop_num 10 \
    --is_llmda True \
    --iter_num 3 \
    --load_history False