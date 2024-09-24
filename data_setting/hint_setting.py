import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from config.api_keys import get_api_key_by_name
from data_setting.utils import CustomGPT4, add_hint_to_dataset
from config.constants import RAW_DATA_DIR_NAME, DATASET_DIR_NAME, LLASA_RESULTS_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME


def main():
    # Set up the argument parser
    parser = ArgumentParser(description="Generate and append hints to datasets using a GPT-4 model.")
    parser.add_argument("--instruction_prompt", type=str, default="You're a teacher creating a relational database exam question. Write indirect hints concisely so that students can easily solve the question based on the question, choices, and answers.", help="Instruction prompt for the GPT-4 model to generate hints.")
    parser.add_argument("--save_name", type=str, default=None, help="Optional name for the output dataset file. If not provided, the original dataset name will be used.")
    parser.add_argument("--train_dataset", type=str, default="dk_train_question.json", help="Filename of the training dataset (JSON format) located in the dataset directory.")
    parser.add_argument( "--valid_dataset", type=str, default=None, help="Filename of the validation dataset (JSON format) located in the dataset directory.")
    parser.add_argument("--test_dataset", type=str, default="dk_test_question.json", help="Filename of the test dataset (JSON format) located in the dataset directory.")
    args = parser.parse_args()

    api_key = get_api_key_by_name('openai')
    if api_key is None:
        print("Error: OpenAI API key not found. Please check your API key configuration.")
        sys.exit(1)

    model = CustomGPT4(api_key)

    # Process each dataset (train, validation, test) if provided
    for dataset, dataset_name in zip(
        [args.train_dataset, args.valid_dataset, args.test_dataset],
        ["train", "validation", "test"]
    ):
        # Construct the full path to the dataset file
        dataset = f"{DATASET_DIR_NAME}/{dataset}" if dataset is not None else None
        if dataset is not None:
            print(f"Processing {dataset_name} dataset: {dataset}")
            try:
                add_hint_to_dataset(model, args.instruction_prompt, dataset, args.save_name)
                print(f"{dataset_name.capitalize()} dataset processed successfully.")
            except Exception as e:
                print(f"Error processing {dataset_name} dataset: {e}")
        else:
            print(f"No {dataset_name} dataset provided. Skipping.")


if __name__ == "__main__":
    main()