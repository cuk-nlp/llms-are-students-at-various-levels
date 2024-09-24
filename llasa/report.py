import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import time
import datasets
from argparse import ArgumentParser
from distutils.util import strtobool

from llasa.framework import Reporter
from config.constants import RAW_DATA_DIR_NAME, DATASET_DIR_NAME, LLASA_RESULTS_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME


def main():
    parser = ArgumentParser(description="LLaSA Evaluation")
    parser.add_argument("--test_dataset", type=str, help="Path to the test dataset in JSON format.")
    parser.add_argument("--llasa_result", type=str, help="CSV.")
    parser.add_argument("--data_type", type=str, default="dk", help="Specify the data type (e.g., 'dk').")
    parser.add_argument("--llasa_type", type=str, default="basic", help="basic or zeroshot.")
    parser.add_argument("--is_llmda", type=strtobool, default=False, help="Whether to use LLMDA.")
    parser.add_argument("--classification_setting_num", type=int, default=6, help="Number of classification bins.")
    args = parser.parse_args()
    args.is_llmda = bool(args.is_llmda)

    # Load dataset and results
    result_df = pd.read_csv(f"{LLASA_RESULTS_DIR_NAME}/{args.llasa_result}")
    test_dataset = datasets.load_dataset("json", data_files=f"{DATASET_DIR_NAME}/{args.test_dataset}")['train'].sort("question_id")

    # Log and print the results
    reporter = Reporter()
    reporter.log_results(result_df, test_dataset, args.llasa_type, args.is_llmda, args.classification_setting_num)


if __name__ == "__main__":
    main()
