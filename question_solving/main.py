import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import copy
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import permutations

from config.api_keys import get_api_key_by_name
from question_solving.utils.dataset_utils import get_dataset_info, get_questions_with_exemplars, load_hf_dataset, load_local_dataset
from question_solving.src.experiment_config import ExperimentConfig
from question_solving.src.experiment_saver import ExperimentSaver
from question_solving.src.models import get_model_by_name


def get_congfig_and_api_key_name_from_args():
    parser = ArgumentParser()
    parser.add_argument("--ds_name", type=str, help="Name of the dataset (ae, ac, rh, rm, dk)")
    parser.add_argument("--prompt_style", type=str, help="Style of the prompt (none, role, fewshot)")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--style_name", type=str, help="Name of the style")
    parser.add_argument("--api_key_name", type=str, help="Name of the API key")
    parser.add_argument("--quantization", type=strtobool, default=False, help="Whether to quantize the model")
    parser.add_argument("--n_shots", type=int, help="Number of shots")
    parser.add_argument("--do_perm", type=strtobool, default=False, help="Whether to permute the exemplars")
    args = parser.parse_args()
    args.quantization = bool(args.quantization)
    args.do_perm = bool(args.do_perm)
    api_key_name = args.api_key_name
    args = vars(args)
    print(args)
    del args["api_key_name"]
    return ExperimentConfig(**args), api_key_name

def run_experiment(config, api_key_name):
    print(config)
    # Get API key
    api_key = get_api_key_by_name(api_key_name)
    
    # Load model
    model = get_model_by_name(
        name = config.model_name, 
        api_key = api_key,
        quantization = config.quantization
    )
    model = {
        "natural": model.process_question_natural,
        "brown": model.process_question_brown,
        "poe": model.process_question_poe,
        "cot": model.process_question_cot,
        "ps": model.process_question_ps,
    }[config.style_name]
    
    # Get questions with exemplars
    qwes = get_questions_with_exemplars(
        info = get_dataset_info(config.ds_name),
        n_shots = config.n_shots,
        style = config.prompt_style,
        load_fn = load_local_dataset if config.ds_name in ["dk", "ast"] else load_hf_dataset
    )
    
    # Run experiment, saving results
    saver = ExperimentSaver(save_fname=config.get_save_fname())
    for q_idx, qwe in enumerate(tqdm(qwes)):
        if config.do_perm:
            for perm_order in permutations(range(qwe.get_n_choices())):
                qwe_copy = copy.deepcopy(qwe)
                qwe_copy.permute_choices(perm_order)
                response = model(qwe_copy)
                saver["question_idx"].append(q_idx)
                saver["perm_order"].append(perm_order)
                saver["qwe"].append(vars(qwe_copy))
                saver["model_response"].append(vars(response))
                
            saver.save()
        else:
            response = model(qwe)
            saver["question_idx"].append(qwe.question_id)
            saver["qwe"].append(vars(qwe))
            saver["model_response"].append(vars(response))
    
    saver.save()


if __name__ == "__main__":
    run_experiment(*get_congfig_and_api_key_name_from_args())            