import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import os
import numpy as np


def make_dir_if_does_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
        
def idx_to_ltr(idx):
    # this is used to convert the index of a choices to a letter (0, 1, 2 -> A, B, C)
    return chr(idx + ord("A"))


def ltr_to_idx(ltr):
    # this is used to convert a letter to the index of a choices (A, B, C -> 0, 1, 2)
    return ord(ltr) - ord("A")


def prep_openai_obj_for_save(obj, prompt_text=None):
    obj = dict(obj)
    for key in obj.keys():
        if isinstance(obj[key], openai.openai_object.OpenAIObject):
            obj[key] = prep_openai_obj_for_save(obj[key])
        if isinstance(obj[key], list):
            for i in range(len(obj[key])):
                if isinstance(obj[key][i], openai.openai_object.OpenAIObject):
                    obj[key][i] = prep_openai_obj_for_save(obj[key][i])
    if prompt_text is not None:
        obj["prompt_text"] = prompt_text
    return obj


def distinguish_prefix(logprobs_dict, choices_char):
    # case 1
    flag = True
    for char in choices_char:
        if f"▁{char}" in logprobs_dict.keys():
            continue
        else:
            flag = False
            break
    if flag:
        return "▁"
    
    # case 2
    flag = True
    for char in choices_char:
        if f"Ġ{char}" in logprobs_dict.keys():
            continue
        else:
            flag = False
            break
    if flag:
        return "Ġ"
    
    # case 3
    flag = True
    for char in choices_char:
        if f" {char}" in logprobs_dict.keys():
            continue
        else:
            flag = False
            break
    if flag:
        return " "
    
    return ""      


def mask_logprobs(token_prefix, logprob_dict, eliminiation_prob_dict):
    for k, v in eliminiation_prob_dict.items():
        if v == -100.0:
            logprob_dict[f"{token_prefix}{k}"] = -np.inf
    
    return logprob_dict