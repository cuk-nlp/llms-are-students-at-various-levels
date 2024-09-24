import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datasets import load_dataset, concatenate_datasets
from config.constants import DATASET_DIR_NAME


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Merge train and test datasets into a single dataset.")
    parser.add_argument('--train_dataset', type=str, default="dk_train_question.json", help="Filename of the training dataset (JSON format).")
    parser.add_argument('--test_dataset', type=str, default="dk_test_question.json", help="Filename of the test dataset (JSON format).")
    args = parser.parse_args()
    
    print("Starting the dataset merging process...")
    
    # Load the training and test datasets
    train_dataset = load_dataset('json', data_files=f"{DATASET_DIR_NAME}/{args.train_dataset}")['train']
    test_dataset = load_dataset('json', data_files=f"{DATASET_DIR_NAME}/{args.test_dataset}")['train']

    # Concatenate the datasets
    merged_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Prepare the output filename by replacing 'train' with 'whole' and save the merged dataset to a JSON file
    merged_dataset_filename = args.train_dataset.replace("train", "whole")
    merged_dataset.to_json(f"{DATASET_DIR_NAME}/{merged_dataset_filename}")
    print(f"Dataset merged and saved to '{merged_dataset_filename}'.")
    

if __name__ == '__main__':
    main()