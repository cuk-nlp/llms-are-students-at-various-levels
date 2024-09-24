import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from distutils.util import strtobool

from src.experiment_config import ExperimentConfig
from config.constants import QUESTION_SOLVING_RESULTS_DIR_NAME
from utils.base_utils import idx_to_ltr


def get_config_from_args():
    parser = ArgumentParser(description="Parse command line arguments for experiment configuration")
    parser.add_argument("--ds_name", help="Name of the dataset being used")
    parser.add_argument("--prompt_style", type=str, help="The format of the prompt (e.g., 'none', 'role', 'desc')")
    parser.add_argument("--model_name", help="Name of the model being evaluated")
    parser.add_argument("--style_name", help="Evaluation style (e.g., 'natural', 'brown')")
    parser.add_argument("--n_shots", type=int, help="Number of examples used for few-shot learning")
    parser.add_argument("--quantization", type=strtobool, default=False, help="Apply model quantization (True/False)")
    parser.add_argument(
        "--do_perm",
        action="store_true",
        default=False,
        help="If set, evaluate all possible permutations of answer choices"
    )

    args = parser.parse_args()
    args.quantization = bool(args.quantization)
    return ExperimentConfig(**vars(args))

def add_correct_answer_col(df):
    df["correct_answer"] = df.apply(
        lambda row: idx_to_ltr(row["qwe"]["answer_idx"]),
        axis=1
    )
    
def div_dicts(a, b):
    # Divide each value in dictionary a by the matching
    # value in dictionary b
    new_dict = dict()
    for key in a.keys():
        if key in b.keys():
            new_dict[key] = a[key] / b[key]
    return new_dict

def sub_dicts(a, b):
    # Subtract from each value in dictionary a the matching
    # value in dictionary b
    new_dict = dict()
    for key in a.keys():
        if key in b.keys():
            new_dict[key] = a[key] - b[key]
    return new_dict

def record_results(config, df):
    total_results_dir = os.path.join(QUESTION_SOLVING_RESULTS_DIR_NAME, "total_results.csv")
    if not os.path.exists(total_results_dir):
        with open(total_results_dir, "w") as f:
            f.write("model_name,style_name,n_shots,do_perm,prompt_style,acc_raw,acc_raw_hard,acc_raw_mid,acc_raw_easy,acc_ln,acc_ln_hard,acc_ln_mid,acc_ln_easy,acc_sn,acc_sn_hard,acc_sn_mid,acc_sn_easy\n")
        f.close()
    total_results_df = pd.read_csv(total_results_dir, encoding="utf-8-sig")
    print(df.columns.tolist())
    
    natural_style_family = ["natural", "poe", "cot", "ps"]
    if config.style_name in natural_style_family:
        new_row = pd.DataFrame({
            'model_name': config.model_name,
            'style_name': config.style_name,
            'n_shots': config.n_shots,
            'do_perm': config.do_perm,
            'prompt_style': config.prompt_style,
            'acc_raw': df["correct"].mean(),
            'acc_ln': None,
            'acc_ln_hard': None,
            'acc_ln_mid': None,
            'acc_ln_easy': None,
            'acc_sn': None,
            'acc_sn_hard': None,
            'acc_sn_mid': None,
            'acc_sn_easy': None,
        }, index=[0])
    elif config.style_name == "brown":
        new_row = pd.DataFrame({
            'model_name': config.model_name,
            'style_name': config.style_name,
            'n_shots': config.n_shots,
            'do_perm': config.do_perm,
            'prompt_style': config.prompt_style,
            'acc_raw': (df["chosen_answer_raw"] == df["correct_answer"]).mean(),
            'acc_ln': (df["chosen_answer_ln"] == df["correct_answer"]).mean(),
            'acc_sn': (df["chosen_answer_sn"] == df["correct_answer"]).mean(),
        }, index=[0])
    total_results_df = pd.concat([total_results_df, new_row], ignore_index=True)
    total_results_df.to_csv(total_results_dir, index=False, encoding="utf-8-sig")
    
def find_model_answer(row, isgpt4=False, isclaude=False):
    if isgpt4:
        sorted_probs = sorted(
            row["model_response"]["logprobs"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        choice_ltrs = [idx_to_ltr(i) for i in range(len(row["qwe"]["choices"]))]
        for (ltr, probs) in sorted_probs:
            if ltr in choice_ltrs: 
                return ltr
        return "ERROR"
    elif isclaude:
        choice_ltrs = [idx_to_ltr(i) for i in range(len(row["qwe"]["choices"]))]
        print(row["model_response"]["response_list"])
        print("#" * 100)
        for ltr in row["model_response"]["response_list"]:
            if ltr in choice_ltrs: 
                return ltr
        return "ERROR"
    else:
        sorted_probs = sorted(
            row["model_response"]["logprobs"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        choice_ltrs = [idx_to_ltr(i) for i in range(len(row["qwe"]["choices"]))]
        for (ltr, probs) in sorted_probs:
            if ltr[1:] in choice_ltrs:
                return ltr[1:] 
        return "ERROR"

def analyze_results(config):
    # Get file name of experiment to load
    fname = config.get_save_fname()

    # Load file
    df = pd.read_pickle(fname)

    natural_style_family = ["natural", "poe", "cot", "ps"]
    if config.style_name in natural_style_family:
        if config.do_perm:
            # We start by calculating the logprob of each
            # answer option irrespective of the order the
            # options were presented in
            def get_lp(ltr, lps):
                if f"▁{ltr}" in lps.keys():
                    return lps[f"▁{ltr}"]
                elif f"Ġ{ltr}" in lps.keys():
                    return lps[f"Ġ{ltr}"]
                elif f" {ltr}" in lps.keys():
                    return lps[f" {ltr}"]
                else:
                    return -np.inf

            df["ord_lps"] = df.apply(
                lambda row: [
                    get_lp(
                        idx_to_ltr(row['perm_order'].index(i)).upper(),
                        row["model_response"]["logprobs"]
                    ) for i in range(len(row["perm_order"]))],
                axis=1
            )

            df["coverage"] = df.apply(
                lambda row: np.sum(np.exp(row["ord_lps"])),
                axis=1
            )
            print(f"Coverage: {df['coverage'].mean()}")

            # Add a column for if model got question right
            df["correct"] = df.apply(
                lambda row: max(
                    row["model_response"]["logprobs"].items(),
                    key=lambda x: x[1]
                    # In line below [0] is the key (as opposed to value)
                    # Additionally we use 1: instead of lstrip because
                    # we want the prediction "A" to be wrong when " A"
                    # is expected, for example
                )[0][1:] == idx_to_ltr(row["qwe"]["answer_idx"]),
                axis=1
            )
            print(f"Accuracy: {df['correct'].mean()}")

            # Making lists of lists
            grouped = df.groupby("question_idx")["ord_lps"].apply(list)
            lps_by_question = grouped.tolist()

            # HOW MANY OF THE CHOSEN ANSWERS MATCH THE MAJORITY
            # ANSWER?
            props = list()
            for q_lps in lps_by_question:
                majority_choice = stats.mode(
                    [np.argmax(x) for x in q_lps]
                )[0][0]
                props.append(
                    sum(
                        [np.argmax(x) == majority_choice for x in q_lps]
                    ) / len(q_lps)
                )

            print("PPA:", np.mean(props))
        else:
            add_correct_answer_col(df)
            if config.model_name == "gpt4" or config.model_name == "gpt4o":
                df["chosen_answer_raw"] = df.apply(
                    lambda row: find_model_answer(row, isgpt4=True),
                    axis=1
                )
            elif "claude" in config.model_name:
                df["chosen_answer_raw"] = df.apply(
                    lambda row: find_model_answer(row, isclaude=True),
                    axis=1
                )
            else:
                df["chosen_answer_raw"] = df.apply(
                    lambda row: find_model_answer(row),
                    axis=1
                )

            df["correct"] = df.apply(
                lambda row: row["chosen_answer_raw"] == row["correct_answer"],
                axis=1
            )

            print(
                "Accuracy:",
                df["correct"].mean()
            )

            # If config.ds_name == "mmlu" we'll present accuracy
            # after grouping by "task"
            if config.ds_name == "mmlu":
                print("Accuracy by task:")
                g = df.groupby("task")["correct"].mean()
                for i, task_name in enumerate(g.index):
                    print(task_name, round(g[i]*100, 1))
    else:
        add_correct_answer_col(df)
        df["chosen_answer_raw"] = df.apply(
            lambda row: max(
                row["model_response"]["logprobs"].items(),
                key=lambda x: x[1]
                # In line below [0] is the key (as opposed to value)
                # No need for 1: here because we assign the letters
                # manually in models.py
            )[0],
            axis=1
        )
        print(
            "Accuracy (raw):",
            (df["chosen_answer_raw"] == df["correct_answer"]).mean()
        )

        # Answer with length normalization
        df["chosen_answer_ln"] = df.apply(
            lambda row: max(
                div_dicts(
                    row["model_response"]["logprobs"],
                    row["model_response"]["lens"]
                ).items(),
                key=lambda x: x[1]
            )[0],
            axis=1
        )
        print(
            "Accuracy (length-normalized):",
            (df["chosen_answer_ln"] == df["correct_answer"]).mean()
        )

        # Answer with special normalization
        df["chosen_answer_sn"] = df.apply(
            lambda row: max(
                sub_dicts(
                    row["model_response"]["logprobs"],
                    row["model_response"]["unconditional_logprobs"]
                ).items(),
                key=lambda x: x[1]
            )[0],
            axis=1
        )
        print(
            "Accuracy (unconditional-normalized):",
            (df["chosen_answer_sn"] == df["correct_answer"]).mean()
        )
        df["chosen_answer_raw_correct"] = df.apply(
            lambda row: int(row["chosen_answer_raw"] == row["correct_answer"]),
            axis=1
        )
        df["chosen_answer_ln_correct"] = df.apply(
            lambda row: int(row["chosen_answer_ln"] == row["correct_answer"]),
            axis=1
        )
        df["chosen_answer_sn_correct"] = df.apply(
            lambda row: int(row["chosen_answer_sn"] == row["correct_answer"]),
            axis=1
        )
        
    df.to_csv(fname.replace(".pkl", ".csv")) # , encoding="utf-8"
    record_results(config, df)


if __name__ == "__main__":
    analyze_results(get_config_from_args())