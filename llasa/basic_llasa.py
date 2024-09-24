import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import ast
import time
from tqdm import tqdm
from argparse import ArgumentParser
from distutils.util import strtobool
from datasets import load_dataset

from llasa.utils import get_model_list, generate_random_string, update_history_df
from llasa.framework import LLaSA
from config.constants import RAW_DATA_DIR_NAME, DATASET_DIR_NAME, LLASA_RESULTS_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME


def main():
    # Set up the argument parser
    parser = ArgumentParser(description="Script for model clustering and evaluation using LLaSA.")
    parser.add_argument("--data_type", type=str, default="dk", help="Type of dataset (e.g., 'dk' for domain knowledge).")
    parser.add_argument("--model_answer_log", type=str, default="model_answer_log.csv", help="Path to the model answer log file.")
    parser.add_argument("--train_dataset", type=str, default="dk_train_question.json", help="Path to the training dataset (in JSON).")
    parser.add_argument("--train_transaction", type=str, default="train_transaction.csv", help="Path to the training transaction CSV file.")
    parser.add_argument("--test_dataset", type=str, default="dk_test_question.json", help="Path to the testing dataset (in JSON).")
    parser.add_argument("--test_transaction", type=str, default="test_transaction.csv", help="Path to the testing transaction CSV file.")
    parser.add_argument("--cluster_size", type=int, default=4, help="Size of the cluster for LLaSA.")
    parser.add_argument("--max_drop_num", type=int, default=10, help="Maximum number of models to randomly drop.")
    parser.add_argument("--is_llmda", type=strtobool, default=False, help="Boolean flag to enable LLMDA (True/False).")
    parser.add_argument("--iter_num", type=int, default=1, help="Number of iterations for LLMDA; must be > 1 if LLMDA is used.")
    parser.add_argument("--load_history", type=strtobool, default=False, help="Boolean flag to load previous history (True/False).")
    args = parser.parse_args()
    
    args.is_llmda = bool(args.is_llmda)
    args.load_history = bool(args.load_history)

    if args.is_llmda and args.iter_num == 1:
        raise ValueError("If using LLMDA, 'iter_num' must be greater than 1.")
    if not args.is_llmda and args.iter_num != 1:
        raise ValueError("If not using LLMDA, 'iter_num' must be 1.")
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=f"{DATASET_DIR_NAME}/{args.train_dataset}")['train'].sort("question_id")
    test_dataset = load_dataset("json", data_files=f"{DATASET_DIR_NAME}/{args.test_dataset}")['train'].sort("question_id")
    
    train_student_response_df = pd.read_csv(f"{RAW_DATA_DIR_NAME}/{args.train_transaction}").sort_values(by=['question_id'])
    test_student_response_df = pd.read_csv(f"{RAW_DATA_DIR_NAME}/{args.test_transaction}").sort_values(by=['question_id'])

    # Load or initialize history
    args.history_dir = f"{LLASA_RESULTS_DIR_NAME}/llasa_{args.data_type}_cluster_{args.cluster_size}_llmda_{args.is_llmda}_iter_{args.iter_num}.csv"
    if args.load_history:
        print("Loading previous history...")
        history_df = pd.read_csv(args.history_dir)
        drop_list = list(map(lambda x: ast.literal_eval(x), history_df['drop_models']))
    else:
        print("Starting with a new history file...")
        history_df = pd.DataFrame()
        history_df.to_csv(args.history_dir, index=False)
        drop_list = []

    # Load model responses
    model_response_df = pd.read_csv(f"{QUESTION_SOLVING_RESULTS_DIR_NAME}/{args.model_answer_log}")
    train_model_response_df = model_response_df[model_response_df['question_id'].isin(train_dataset['question_id'])].reset_index(drop=True)
    test_model_response_df = model_response_df[model_response_df['question_id'].isin(test_dataset['question_id'])].reset_index(drop=True)

    # Get the list of models
    model_list = get_model_list(model_response_df)

    # Initialize LLaSA framework
    llasa = LLaSA()

    # Begin the LLaSA process
    print("Starting the LLaSA process...\n")
    for current_iter in tqdm(range(args.iter_num)):
        time.sleep(0.05)
        
        # LLMDA - Randomly drop models
        if args.is_llmda:
            drop_num = random.randint(1, args.max_drop_num)
        else:
            drop_num = 0
        drop_models = set(np.random.choice(model_list, drop_num))
        iter_model_list = [model for model in model_list if model not in drop_models]

        # Skip if this drop combination was used before
        if drop_models in drop_list:
            continue
        drop_list.append(drop_models)

        # Perform IRT on selected models
        selected_train_model = [col for col in train_model_response_df.columns if any(model in col for model in iter_model_list)]
        question_id_df = train_model_response_df['question_id']
        temp_train_model_response_df = pd.concat([question_id_df, train_model_response_df[selected_train_model]], axis=1)
        
        model_ability_dict, model_difficulty_dict = llasa.get_irt(temp_train_model_response_df, LLASA_RESULTS_DIR_NAME, "trash", generate_random_string(100), is_remove=True)
        
        # Cluster student responses
        student_model_pair_list = llasa.student_representative_llm_cluster_selection(
            train_student_response_df, train_dataset['ability'][0], model_ability_dict, args.cluster_size
        )

        # Aggregate responses for training and testing datasets
        train_student_aggregation_result_df = llasa.llm_cluster_response_aggregation(student_model_pair_list, train_model_response_df)
        train_student_aggregation_result_df['question_id'] = train_student_response_df['question_id']

        test_student_aggregation_result_df = llasa.llm_cluster_response_aggregation(student_model_pair_list, test_model_response_df)
        test_student_aggregation_result_df['question_id'] = test_student_response_df['question_id']

        # Evaluate aggregated responses using IRT and calculate metrics
        _, train_model_difficulty_dict = llasa.get_irt(train_student_aggregation_result_df, LLASA_RESULTS_DIR_NAME, "trash", generate_random_string(10), is_remove=True)
        train_mse, train_rmse, train_mae, train_corr, train_corr_p = llasa.calculate_metrics(
            train_dataset['pred_irt'], list(train_model_difficulty_dict.values())
        )

        _, test_model_difficulty_dict = llasa.get_irt(test_student_aggregation_result_df, LLASA_RESULTS_DIR_NAME, "trash", generate_random_string(10), is_remove=True)
        test_mse, test_rmse, test_mae, test_corr, test_corr_p = llasa.calculate_metrics(
            test_dataset['pred_irt'], list(test_model_difficulty_dict.values())
        )

        # Log the results and update history
        new_row = {
            'data_type': args.data_type,
            'is_llmda': args.is_llmda,
            'cluster_size': args.cluster_size,
            'iter_num': current_iter,
            'max_drop_num': args.max_drop_num,
            'model_answer_log_dir': f"{QUESTION_SOLVING_RESULTS_DIR_NAME}/{args.model_answer_log}",
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_corr': train_corr,
            'train_corr_p': train_corr_p,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_corr': test_corr,
            'test_corr_p': test_corr_p,
            'train_models': model_ability_dict,
            'test_models': test_model_difficulty_dict,
            "train_difficulty": train_model_difficulty_dict,
            "test_difficulty": test_model_difficulty_dict,
            "student_model_pair_list": student_model_pair_list,
            'iter_model_list': iter_model_list,
            'drop_num': drop_num,
            'drop_models': drop_models,
        }
        history_df = update_history_df(history_df, args.history_dir, new_row, "llasa")

    print(f"\nProcess complete! The updated result has been saved to {args.history_dir}.")

    
if __name__ == "__main__":
    main()
