#!/bin/bash

python data_setting/irt_setting.py \
  --data_type dk \
  --train_type train \
  --question_dataset train_question.json \
  --transaction_dataset train_transaction.csv

python data_setting/irt_setting.py \
  --data_type dk \
  --train_type test \
  --question_dataset test_question.json \
  --transaction_dataset test_transaction.csv