import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from argparse import ArgumentParser
from distutils.util import strtobool
from datasets import load_dataset

from llasa.utils import get_model_list, generate_random_string, update_history_df
from llasa.framework import ZeroshotLLaSA
from config.constants import RAW_DATA_DIR_NAME, DATASET_DIR_NAME, LLASA_RESULTS_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME


def main():
    # Set up the argument parser
    parser = ArgumentParser(description="Zero-shot LLaSA: Clustering and Evaluation")
    parser.add_argument("--data_type", type=str, default="dk", help="Specify the data type (e.g., 'dk', 'ast').")
    parser.add_argument("--model_answer_log", type=str, default="model_answer_log.csv", help="File name for the model answer log.")
    parser.add_argument("--train_dataset", type=str, default="dk_train_question.json", help="File name for the training dataset.")
    parser.add_argument("--test_dataset", type=str, default="dk_test_question.json", help="File name for the test dataset.")
    parser.add_argument("--low_level_llms", type=int, default=150, help="Number of low-level LLMs to use.")
    parser.add_argument("--middle_level_llms", type=int, default=60, help="Number of middle-level LLMs to use.")
    parser.add_argument("--high_level_llms", type=int, default=30, help="Number of high-level LLMs to use.")
    parser.add_argument("--load_history", type=strtobool, default=False, help="Boolean flag to load previous history (True/False).")
    args = parser.parse_args()
    args.load_history = bool(args.load_history)
    
    # Map LLM levels to their corresponding counts
    args.level_llms = {
        "low": args.low_level_llms,
        "middle": args.middle_level_llms,
        "high": args.high_level_llms
    }
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=f"{DATASET_DIR_NAME}/{args.train_dataset}")['train'].sort("question_id")
    test_dataset = load_dataset("json", data_files=f"{DATASET_DIR_NAME}/{args.test_dataset}")['train'].sort("question_id")
    
    # Load or initialize history
    args.history_dir = f"{LLASA_RESULTS_DIR_NAME}/zeroshot_llasa_{args.data_type}_low_{args.low_level_llms}_middle_{args.middle_level_llms}_high_{args.high_level_llms}.csv"
    if args.load_history:
        print(f"Loading previous history from {args.history_dir}...")
        history_df = pd.read_csv(args.history_dir)
    else:
        print("Starting a new history file...")
        history_df = pd.DataFrame()
        history_df.to_csv(args.history_dir, index=False)

    # Load model responses and calculate average accuracy for each model
    model_response_df = pd.read_csv(f"{QUESTION_SOLVING_RESULTS_DIR_NAME}/{args.model_answer_log}")
    train_model_response_df = model_response_df[model_response_df['question_id'].isin(train_dataset['question_id'])].reset_index(drop=True)
    test_model_response_df = model_response_df[model_response_df['question_id'].isin(test_dataset['question_id'])].reset_index(drop=True)

    train_model_accuracy_avg_dict = train_model_response_df.drop(columns=['question_id']).mean().to_dict()
    train_model_accuracy_avg_df = pd.DataFrame(train_model_accuracy_avg_dict.items(), columns=['key', 'value'])

    # Initialize the Zero-shot LLaSA model
    zeroshot_llasa = ZeroshotLLaSA(train_model_accuracy_avg_df, args.level_llms)
    
    # Begin the Zero-shot LLaSA process
    print("Starting the Zero-shot LLaSA process...")
    
    sampled_models_list = zeroshot_llasa.llm_selection()

    # Evaluate the selected models on the training dataset
    temp_train_model_response_df = train_model_response_df[['question_id'] + sampled_models_list]
    train_model_ability_dict, train_model_difficulty_dict = zeroshot_llasa.get_irt(
        temp_train_model_response_df, LLASA_RESULTS_DIR_NAME, "trash", generate_random_string(100), is_remove=True
    )
    train_mse, train_rmse, train_mae, train_corr, train_corr_p = zeroshot_llasa.calculate_metrics(
        train_dataset['pred_irt'], list(train_model_difficulty_dict.values())
    )

    # Evaluate the selected models on the test dataset
    temp_test_model_response_df = test_model_response_df[['question_id'] + sampled_models_list]
    test_model_ability_dict, test_model_difficulty_dict = zeroshot_llasa.get_irt(
        temp_test_model_response_df, LLASA_RESULTS_DIR_NAME, "trash", generate_random_string(100), is_remove=True
    )
    test_mse, test_rmse, test_mae, test_corr, test_corr_p = zeroshot_llasa.calculate_metrics(
        test_dataset['pred_irt'], list(test_model_difficulty_dict.values())
    )

    # Log the results and update history
    new_row = {
        'data_type': args.data_type,
        "low_level_llms": args.low_level_llms,
        "middle_level_llms": args.middle_level_llms,
        "high_level_llms": args.high_level_llms,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_corr": train_corr,
        "train_corr_p": train_corr_p,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_corr": test_corr,
        "test_corr_p": test_corr_p,
        "sampled_models_list": sampled_models_list,
        'train_models': train_model_ability_dict,
        'test_models': test_model_difficulty_dict,
        "train_difficulty": train_model_difficulty_dict,
        "test_difficulty": test_model_difficulty_dict,
    }
    history_df = update_history_df(history_df, args.history_dir, new_row, "zeroshot_llasa")
    
    print(f"Process complete! The updated result has been saved to {args.history_dir}.")


if __name__ == "__main__":
    main()