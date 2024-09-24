import random
import string
import pandas as pd


def get_model_list(model_response_df):
    model_list = []

    for i in range(1, len(model_response_df.columns)): # Skip the first column, which is the question ID
        col_parts = model_response_df.columns[i].split('_')[3:]

        stop_idx = len(col_parts)
        for j, part in enumerate(col_parts):
            if part in ['cot', 'ps', 'natural', 'poe']:
                stop_idx = j
                break

        model_name = "_".join(col_parts[:stop_idx])
        model_list.append(model_name)

    return list(set(model_list))


def generate_random_string(length):
    letters = string.ascii_letters  # Contains both lowercase and uppercase letters
    return ''.join(random.choice(letters) for _ in range(length))


def update_history_df(history_df, history_dir, new_row, method):
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv(history_dir, index=False)
        
    return history_df
