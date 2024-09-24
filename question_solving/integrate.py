import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd
from argparse import ArgumentParser

from config.constants import RAW_DATA_DIR_NAME, DATASET_DIR_NAME, LLASA_RESULTS_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME


def main():
    # Set up the argument parser    
    parser = ArgumentParser(description="Process dataset and log file to create a consolidated results log")
    parser.add_argument("--whole_dataset", type=str, help="Name of the complete dataset file (JSON format)")
    parser.add_argument("--log_name", type=str, default="", help="Name of the folder containing result log files")
    args = parser.parse_args()
    
    data = []
    dataset_path = os.path.join(DATASET_DIR_NAME, args.whole_dataset)
    for line in open(dataset_path, "r"):
        data.append(json.loads(line))
    
    log_dir = os.path.join(QUESTION_SOLVING_RESULTS_DIR_NAME, args.log_name)
    log_files = os.listdir(log_dir)
    
    log_files = [f for f in log_files if f.endswith(".csv")]
    if "total_results.csv" in log_files:
        log_files.remove("total_results.csv")
    if "model_answer_log.csv" in log_files:
        log_files.remove("model_answer_log.csv")
    
    total_log = pd.DataFrame([single_data["question_id"] for single_data in data], columns=['question_id'])
    
    for log_name in log_files:
        single_log_path = os.path.join(log_dir, log_name)
        single_log = pd.read_csv(single_log_path)
        try:
            correct_series = single_log["correct"].astype(int)  # Ensure the 'correct' column is properly formatted
        except:
            print(f"Error processing {log_name}. Please check the file format.")
            raise AssertionError("Failed to convert 'correct' column to integer format.")
        
        total_log[log_name] = correct_series
    
    output_path = os.path.join(log_dir, "model_answer_log.csv")
    total_log.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Consolidated results saved to {output_path}")


if __name__ == "__main__":
    main()