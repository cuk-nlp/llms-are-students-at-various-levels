import openai
import time
from tqdm import tqdm
import datasets

from config.constants import GPT4_MODEL_NAME, RETRY_SLEEP_TIME


class CustomOpenAIModel:
    def __init__(self, api_key, model_name, add_space=False):
        self.client = openai.OpenAI(api_key=api_key)
        self.add_space = add_space
        self.model_name = model_name

    def process_question_natural(self, prompt_text):
        response = self._get_response(prompt_text)
        return response.choices[0].message.content

    def _get_response(self, text):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": text + (" " if self.add_space else "")}
                    ],
                    temperature=0,  # Use temperature 0 for deterministic responses
                    max_tokens=1024
                )
                return response
            except Exception as e:
                print(f"API call error: {e}")
                print(f"Retrying after {RETRY_SLEEP_TIME} seconds...")
                time.sleep(RETRY_SLEEP_TIME)

    def process_question_brown(self, question):
        print("The 'echo' parameter is not supported in this method.")
        raise NotImplementedError


class CustomGPT4(CustomOpenAIModel):
    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT4_MODEL_NAME)


def add_hint_to_dataset(model, instruction_prompt, dataset_dir, save_name):
    ds = datasets.load_dataset("json", data_files=dataset_dir)['train']
    response_list = []
    choices_list = ["A", "B", "C", "D", "E", "F", "G"]  # Adjust according to your dataset

    # Process each example in the dataset
    for i in tqdm(range(len(ds)), desc="Processing dataset"):
        example = ds[i]
        instruction_text = f"Instruction: {instruction_prompt}\n\n"
        question_text = f"Question: {example['question_text']}\n"
        choice_text = ""
        hint_text = "\n\nHint:"
        response_text = ""

        # Build the choice text and identify the correct answer
        for idx, choice in enumerate(example['choices']):
            choice_text += f"{choices_list[idx]}. {choice}\n"
            if choice == example['answer'][0]:
                response_text = f"({choices_list[idx]}) {choice}"

        # Construct the input text for the model
        input_text = instruction_text + question_text + choice_text + "Answer: "
        input_text_with_response = input_text + response_text + hint_text

        # Get the model's response
        response = model.process_question_natural(input_text_with_response)
        response_list.append(response)

    # Add the generated hints as a new column to the dataset
    ds = ds.add_column("hint", response_list)

    # Determine the save path for the updated dataset
    if save_name is not None:
        dataset_save_path = dataset_dir.split(".")[0] + f"_{save_name}.json"
    else:
        dataset_save_path = dataset_dir

    # Save the updated dataset with hints
    ds.to_json(dataset_save_path)
    print(f"Dataset with hints saved at: {dataset_save_path}")