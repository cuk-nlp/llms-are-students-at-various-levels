import os

# API model names
GPT3_BABBAGE_MODEL_NAME = "babbage-002"
GPT3_DAVINCI_MODEL_NAME = "davinci-002"
GPT3_5_MODEL_NAME = "gpt-3.5-turbo-0125"
GPT4_MODEL_NAME = "gpt-4-turbo"
GPT4O_MODEL_NAME = "gpt-4o"
CLAUDE3_HAIKU_MODEL_NAME = "claude-3-haiku-20240307"
CLAUDE3_SONNET_MODEL_NAME = "claude-3-sonnet-20240229"
CLAUDE3_OPUS_MODEL_NAME = "claude-3-opus-20240229"
PALM2_MODEL_NAME = ""
GEMINI_MODEL_NAME = ""

# Other constants
RETRY_SLEEP_TIME = 10
REPRODUCIBILITY_SEED = 42

# Directory paths
HF_CACHE_DIR_NAME = "/data/.cache/huggingface"  # Hugging Face model cache directory
RAW_DATA_DIR_NAME = './data/raw'  # Directory to save processed datasets
DATASET_DIR_NAME = './data/processed'  # Directory to save processed datasets
LLASA_RESULTS_DIR_NAME = "./logs/llasa"
QUESTION_SOLVING_RESULTS_DIR_NAME = "./logs/question_solving"  # Directory to save question-solving results

# API tokens
WANDB_TOKEN = "WRITE_YOUR_WAND_TOKEN_HERE"  # Weights & Biases token
HF_TOKEN = "WRITE_YOUR_HF_TOKEN_HERE"

def make_dir_if_does_not_exist(dir_name):
    """Creates the directory if it does not already exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Ensure the specified directories exist
for dir_name in [HF_CACHE_DIR_NAME, DATASET_DIR_NAME, QUESTION_SOLVING_RESULTS_DIR_NAME, LLASA_RESULTS_DIR_NAME]:
    make_dir_if_does_not_exist(dir_name)