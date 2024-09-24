#!/bin/bash

model_list="llama_2_70b llama_2_70b_chat llama_3_8b llama_3_8b_instruct llama_3_70b llama_3_70b_instruct vicuna_1_33b mixtral mixtral_chat flan_t5_small flan_t5_base flan_t5_large flan_t5_xl flan_t5_xxl upstage_llama_1_30b upstage_llama_1_65b upstage_llama_2_70b solar_70b gpt_neo_125m gpt_neo_1.3b gpt_neo_2.7b gpt_neox_20b gpt_j_6b opt_125m opt_350m opt_1.3b opt_2.7b yi_34b_chat falcon_40b falcon_40b_instruct pythia_410m pythia_1b pythia_1.4b pythia_2.8b pythia_6.9b pythia_12b mistral mistral_chat llama_2_7b llama_2_13b llama_2_7b_chat llama_2_13b_chat vicuna_1_7b vicuna_1_13b vicuna_2_7b vicuna_2_13b solar_10.7b solar_10.7b_instruct yi_6b  yi_6b_chat amber amber_chat crystal_coder crystal_chat falcon_7b falcon_7b_instruct solar_orcadpo_solar_instruct_slerp openchat openchat_2 openchat_2_w openchat_3.2 openchat_3.2_super openchat_3.5 starling orca_2_7b orca_2_13b zephyr_alpha zephyr_beta claude3_sonnet claude3_opus gpt3.5 gpt4 gpt4o"
shot_list="0 1 3 5 10 20 30"
style_list="natural cot ps"

for model_name in $model_list
do
    for style_name in $style_list
    do  
        for shot in $shot_list
        do
            python question_solving/analyze.py \
            --ds_name="dk" \
            --prompt_style="hint_info" \
            --model_name=$model_name \
            --style_name=$style_name \
            --quantization=true \
            --n_shots=$shot
        done
    done
done