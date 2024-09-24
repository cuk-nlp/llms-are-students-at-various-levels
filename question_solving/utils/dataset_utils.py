import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from dataclasses import dataclass
from datasets import load_dataset
from typing import Callable

from config.constants import HF_CACHE_DIR_NAME, REPRODUCIBILITY_SEED
from src.prompts import Exemplar, QuestionPart, QuestionWithExemplar
from utils.base_utils import idx_to_ltr, ltr_to_idx


@dataclass
class DatasetInfo:
    path: str
    exemplar_split: str
    eval_split: str
    extractor: Callable
    name: str = None
    data_dir: str = None
    
    
def load_hf_dataset(path, name, data_dir, split):
    return load_dataset(
        path=path,
        name=name,
        data_dir=data_dir,
        split=split,
        cache_dir=HF_CACHE_DIR_NAME
    )
    
    
def load_local_dataset(train=None, test=None, whole=None):
    data_files = {
        "train": train,
        "test": test,
        "whole": whole
    }
    print(data_files)
    return load_dataset(
            "json",
            data_files=data_files
        )
    
    
def get_questions_with_exemplars(
    info,
    n_shots,
    style,
    load_fn=load_local_dataset,
):        
    if info.exemplar_split == info.eval_split:
        if info.name == "DBE-KT22" and info.exemplar_split == "whole":
            pass
        elif info.name == "ASSISTments" and info.exemplar_split == "whole":
            pass
        else:
            raise ValueError("Exemplar and eval split must be different!")
    
    base_prompt = ""
    if info.name == "DBE-KT22":
        if n_shots==0:
            if style == "none":
                base_prompt = ""   
            elif style == "role":
                base_prompt = "You are a intelligent agent specialized for database subject problem solving. Read the questions and options below, understand the question and select one answer from the choices."
            elif style == "hint":
                base_prompt = "You are an intelligent agent specialized for database subject problem solving. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems."
            elif style == "info":
                base_prompt = "You are an intelligent agent specialized for database subject problem solving. The question below is about relational databases as taught at the Australian National University. The exam is intended for undergraduate and postgraduate students with a variety of majors, including computer science, engineering, arts, and business. Given the diversity of students' majors and learning experiences, the difficulty level of the exam will vary depending on the students' background and understanding of relational databases. The content is likely to be relatively familiar to computer science and engineering majors, but may be more challenging for arts or business majors. Therefore, the difficulty of the exam will vary depending on the student's major and relevant experience. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices."
            elif style == "hint_info":
                base_prompt = "You are an intelligent agent specialized for database subject problem solving. The question below is about relational databases as taught at the Australian National University. The exam is intended for undergraduate and postgraduate students with a variety of majors, including computer science, engineering, arts, and business. Given the diversity of students' majors and learning experiences, the difficulty level of the exam will vary depending on the students' background and understanding of relational databases. The content is likely to be relatively familiar to computer science and engineering majors, but may be more challenging for arts or business majors. Therefore, the difficulty of the exam will vary depending on the student's major and relevant experience. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems."
            else:
                raise ValueError(f"Invalid style: {style}")
        else:
            if style == "none":
                base_prompt = ""   
            elif style == "role":
                base_prompt = "You are a intelligent agent specialized for database subject problem solving. Read the questions and options below, understand the question and select one answer from the choices."
            elif style == "hint":
                base_prompt = "You are an intelligent agent specialized for database subject problem solving. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems."
            elif style == "info":
                base_prompt = "You are an intelligent agent specialized for database subject problem solving. The question below is about relational databases as taught at the Australian National University. The exam is intended for undergraduate and postgraduate students with a variety of majors, including computer science, engineering, arts, and business. Given the diversity of students' majors and learning experiences, the difficulty level of the exam will vary depending on the students' background and understanding of relational databases. The content is likely to be relatively familiar to computer science and engineering majors, but may be more challenging for arts or business majors. Therefore, the difficulty of the exam will vary depending on the student's major and relevant experience. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices. Example of the question and desired answer is like below."
            elif style == "hint_info":
                base_prompt = "You are an intelligent agent specialized for database subject problem solving. The question below is about relational databases as taught at the Australian National University. The exam is intended for undergraduate and postgraduate students with a variety of majors, including computer science, engineering, arts, and business. Given the diversity of students' majors and learning experiences, the difficulty level of the exam will vary depending on the students' background and understanding of relational databases. The content is likely to be relatively familiar to computer science and engineering majors, but may be more challenging for arts or business majors. Therefore, the difficulty of the exam will vary depending on the student's major and relevant experience. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems. Example of the question and desired answer is like below."            
            else:
                raise ValueError(f"Invalid style: {style}")
    elif info.name == "ASSISTments":
        if n_shots==0:
            if style == "none":
                base_prompt = ""   
            elif style == "role":
                base_prompt = "You are a intelligent agent specialized for various subject problem solving. Read the questions and options below, understand the question and select one answer from the choices."
            elif style == "hint":
                base_prompt = "You are an intelligent agent specialized for various subject problem solving. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems."
            elif style == "info":
                base_prompt = "You are an intelligent agent specialized for various subject problem solving. The question below is a rich educational dataset derived from the ASSISTments online tutoring system, which is used to help students with math and other subjects. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices."
            elif style == "hint_info":
                base_prompt = "You are an intelligent agent specialized for various subject problem solving. The question below is a rich educational dataset derived from the ASSISTments online tutoring system, which is used to help students with math and other subjects. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems."
            else:
                raise ValueError(f"Invalid style: {style}")
        else:
            if style == "none":
                base_prompt = ""   
            elif style == "role":
                base_prompt = "You are a intelligent agent specialized for various subject problem solving. Read the questions and options below, understand the question and select one answer from the choices."
            elif style == "hint":
                base_prompt = "You are an intelligent agent specialized for various subject problem solving. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems."
            elif style == "info":
                base_prompt = "You are an intelligent agent specialized for various subject problem solving. The question below is a rich educational dataset derived from the ASSISTments online tutoring system, which is used to help students with math and other subjects. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices. Example of the question and desired answer is like below."
            elif style == "hint_info":
                base_prompt = "You are an intelligent agent specialized for various subject problem solving. The question below is a rich educational dataset derived from the ASSISTments online tutoring system, which is used to help students with math and other subjects. You'll need to step into the role of these students. Read the questions and options below, understand the question and select one answer from the choices. Use any hints provided to assist in solving the problems. Example of the question and desired answer is like below."            
            else:
                raise ValueError(f"Invalid style: {style}")
    else:
        print(f"Specialized base prompt not implemented. Dataset name: {info.name}")
        
    # Create exemplars
    if "hint" in style:
        use_hint = True
    else:
        use_hint = False
    print("use_hint: ", use_hint)
    
    if info.name == "DBE-KT22":
        exemplar_ds = load_fn(
            train=info.path + "dk_train_question.json",
            test=info.path + "dk_test_question.json",
            whole=info.path + "dk_whole_question.json",
        )
        exemplar_ds = exemplar_ds[info.exemplar_split]
    elif info.name == "ASSISTments":
        exemplar_ds = load_fn(
            train=info.path + "ast_train_question.json",
            test=info.path + "ast_test_question.json",
            whole=info.path + "ast_whole_question.json",
        )
        exemplar_ds = exemplar_ds[info.exemplar_split]
    else: 
        exemplar_ds = load_fn(
            path=info.path,
            name=info.name,
            data_dir=info.data_dir,
            split=info.exemplar_split
        )
    exemplars = [Exemplar(**info.extractor(row), **{"use_hint": use_hint}) for row in exemplar_ds]

    if info.name == "DBE-KT22":
        eval_ds = load_fn(
            train=info.path + "dk_train_question.json",
            test=info.path + "dk_test_question.json",
            whole=info.path + "dk_whole_question.json",
        )
        eval_ds = eval_ds[info.eval_split]
    elif info.name == "ASSISTments":
        eval_ds = load_fn(
            train=info.path + "dk_train_question.json",
            test=info.path + "ast_test_question.json",
            whole=info.path + "ast_whole_question.json",
        )
        eval_ds = eval_ds[info.eval_split]
    else:
        eval_ds = load_fn(
            path=info.path,
            name=info.name,
            data_dir=info.data_dir,
            split=info.eval_split
        )
    
    random.seed(REPRODUCIBILITY_SEED)
    qwes = list()
    for row_idx, row in enumerate(eval_ds):
        possible_idxs = list(range(len(exemplars)))
        del possible_idxs[row_idx]
        row_exemplars = [
            exemplars[i] for i in random.sample(possible_idxs, n_shots)
        ]
        row_qwe = QuestionWithExemplar(
            **{**info.extractor(row), **{"exemplar": row_exemplars, "base_prompt": base_prompt, "use_hint": use_hint}}
        )
        qwes.append(row_qwe)
    return qwes


def get_dataset_info(ds_name):
    return {
        "dk": DatasetInfo(
            path="data/processed/",
            name="DBE-KT22",
            exemplar_split="whole",
            eval_split="whole",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question_text"], tag="Question")
                ],
                "choices": row["choices"],
                "answer_idx": row["choices"].index(row["answer"][0]),
                "hint": row["hint"],
                "question_id": row["question_id"]
            }
        ),
        "ast": DatasetInfo(
            path="data/processed/",
            name="ASSISTments",
            exemplar_split="whole",
            eval_split="whole",
            extractor=lambda row: {
                "parts": [
                    QuestionPart(text=row["question_text"], tag="Question")
                ],
                "choices": row["choices"],
                "answer_idx": row["choices"].index(row["answer"][0]),
                "hint": row["hint"],
                "question_id": row["question_id"]
            }
        ),
    }[ds_name]