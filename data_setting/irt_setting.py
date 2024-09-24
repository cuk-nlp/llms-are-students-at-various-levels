import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datasets import load_dataset
from llasa.framework import IRTFramework
from config.constants import RAW_DATA_DIR_NAME, DATASET_DIR_NAME, LLASA_RESULTS_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate an IRT model and append predictions to the dataset.")
    parser.add_argument('--data_type', type=str, default="dk", help="Type of data to process (e.g., 'dk', 'ast').")
    parser.add_argument('--train_type', type=str, default="train", help="Dataset split to process (e.g., 'train', 'val', 'test').")
    parser.add_argument('--question_dataset', type=str, default="train_question.json", help="Filename of the question dataset (JSON format).")
    parser.add_argument('--transaction_dataset', type=str, default="train_transaction.csv", help="Filename of the transaction dataset (CSV format).")
    args = parser.parse_args()

    # Load the question dataset and transaction dataset
    question_dataset = load_dataset('json', data_files=f"{RAW_DATA_DIR_NAME}/{args.question_dataset}")['train']
    transaction_dataset = pd.read_csv(f"{RAW_DATA_DIR_NAME}/{args.transaction_dataset}")

    # Generate IRT difficulty and ability scores
    print("Calculating student abilities and question difficulties using the IRT model...")
    irt = IRTFramework()
    ability_scores, difficulty_scores = irt.get_irt(transaction_dataset, DATASET_DIR_NAME, args.data_type, args.train_type)
        
    student_list = transaction_dataset.columns[1:].tolist() # Skip the 'question_id' column
    student_abilities = {student: ability for ability, student in zip(ability_scores.values(), student_list)}
    
    transaction_dataset['question_acc_mean'] = transaction_dataset.iloc[:, 1:].mean(axis=1)
    transaction_dataset['question_diff'] = transaction_dataset.apply(lambda x: difficulty_scores[x['question_id']], axis=1) 

    def add_irt_predictions(example):
        question_id = example['question_id']
        example['accuracy'] = [transaction_dataset.loc[transaction_dataset['question_id'] == question_id, 'question_acc_mean'].values[0]]
        example['pred_irt'] = transaction_dataset.loc[transaction_dataset['question_id'] == question_id, 'question_diff'].values[0]
        example['ability'] = student_abilities
        return example

    print("Appending IRT predictions to the question dataset...")
    question_dataset = question_dataset.map(add_irt_predictions)

    # Save the updated dataset with IRT predictions
    question_output_path = os.path.join(DATASET_DIR_NAME, f'{args.data_type}_{args.train_type}_question.json')
    question_dataset.to_json(question_output_path)
    print(f"Process complete! The updated question dataset has been saved to '{question_output_path}'.")


if __name__ == '__main__':
    main()