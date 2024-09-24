#!/bin/bash

python data_setting/hint_setting.py \
  --instruction_prompt "You're a teacher creating an exam question. Write a hint concisely so that students can easily solve the question based on the question, choices, and answers." \
  --train_dataset dk_train_question.json \
  --test_dataset dk_test_question.json