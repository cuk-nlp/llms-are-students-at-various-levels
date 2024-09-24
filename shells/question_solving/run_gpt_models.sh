#!/bin/bash

model_list="gpt3.5 gpt4 gpt4o"
shot_list="0 1 3 5 10 20 30"
style_list="natural cot ps"

for shot in $shot_list
do
    for model_name in $model_list
    do
        for style_name in $style_list
        do  
            python scripts/main.py \
            --ds_name="dk" \
            --prompt_style="hint_info" \
            --model_name=$model_name \
            --style_name=$style_name \
            --api_key_name="openai" \
            --quantization=true \
            --n_shots=$shot
        done
    done
done