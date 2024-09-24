import os
import numpy as np
import torch
import torch.nn as nn
import openai
import anthropic
import requests
import time
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig

from config.constants import (
    GPT3_BABBAGE_MODEL_NAME,
    GPT3_DAVINCI_MODEL_NAME,
    GPT3_5_MODEL_NAME,
    GPT4_MODEL_NAME,
    GPT4O_MODEL_NAME,
    CLAUDE3_HAIKU_MODEL_NAME,
    CLAUDE3_SONNET_MODEL_NAME,
    CLAUDE3_OPUS_MODEL_NAME,
    PALM2_MODEL_NAME,
    GEMINI_MODEL_NAME,
    HF_CACHE_DIR_NAME,
    QUESTION_SOLVING_RESULTS_DIR_NAME,
    RETRY_SLEEP_TIME,
    WANDB_TOKEN,
    HF_TOKEN
)
os.system(f"huggingface-cli login --token {HF_TOKEN}")
from utils.base_utils import idx_to_ltr, prep_openai_obj_for_save, distinguish_prefix, mask_logprobs


@dataclass
class ModelResponseNatural:
    logprobs: dict
    response_list: list
    

@dataclass
class ModelResponseBrown:
    logprobs: dict
    unconditional_logprobs: dict
    lens: dict
    response_list: list
    
    
class GPT2Model:
    def __init__(self, model_name, quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cahce_dir=HF_CACHE_DIR_NAME,
            device_map = 'auto'
        )
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                quantization_config=bnb_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto'
            )
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        outputs = self.model(**inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
    def process_question_poe(self, question, strat="below_average"):
        prompt_text_elimination = question.get_natural_prompt()
        inputs = self.tokenizer(prompt_text_elimination, return_tensors="pt", return_token_type_ids=False)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        outputs = self.model(**inputs)
        # output tensor들 디바이스 찍어보고 바꾸는 코드 작성해야 할듯
        for k, v in outputs.items():
            outputs[k] = v.to('cpu')
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)
        
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            ) # [:50]
        }
        
        choices_char = [idx_to_ltr(i) for i in range(question.get_n_choices())]
        token_prefix = distinguish_prefix(logprobs_dict, choices_char)
        
        eliminiation_prob_dict = dict()
        for char in choices_char:
            eliminiation_prob_dict[char] = logprobs_dict[f"{token_prefix}{char}"]
                    
        if strat == "below_average":
            average_prob = 0.0
            for char in choices_char:
                average_prob += eliminiation_prob_dict[char]
            average_prob /= len(choices_char)
            
            for char in choices_char:
                if eliminiation_prob_dict[char] < average_prob:
                    eliminiation_prob_dict[char] = -100.0
        elif strat == "min":
            for i, char in enumerate(choices_char):
                if i == 0:
                    min_prob = eliminiation_prob_dict[char]
                else:
                    if eliminiation_prob_dict[char] < min_prob:
                        min_prob = eliminiation_prob_dict[char]

            for char in choices_char:
                if eliminiation_prob_dict[char] == min_prob:
                    eliminiation_prob_dict[char] = -100.0
        
        prompt_text_prediction = question.get_poe_prompt(eliminiation_prob_dict)
        inputs = self.tokenizer(prompt_text_prediction, return_tensors="pt", return_token_type_ids=False)
        
        outputs = self.model(**inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)
        
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }             
 
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }
        logprobs_dict = mask_logprobs(token_prefix, logprobs_dict, eliminiation_prob_dict)
        
        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0][-(choice_n_tokens):]
            print(f"answer_tokens: {self.tokenizer.decode(answer_tokens)}")
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )

    def process_question_cot(self, question):
        prompt_text = question.get_cot_prompt()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'])
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is "     
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        for k, v in second_inputs.items():
            second_inputs[k] = v.to(self.model.device)
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
    def process_question_ps(self, question):
        prompt_text = question.get_ps_prompt()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'])
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is "
        
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        for k, v in second_inputs.items():
            second_inputs[k] = v.to(self.model.device)
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        

# checked - not used: reporting nan logprobs
# class pythia_70m(GPT2Model):
#     def __init__(self, quantization):
#         super().__init__(model_name="EleutherAI/pythia-70m-deduped", quantization=quantization)


# checked - not used: reporting nan logprobs
# class pythia_160m(GPT2Model):
#     def __init__(self, quantization):
#         super().__init__(model_name="EleutherAI/pythia-160m-deduped", quantization=quantization)
        
    
# checked    
class pythia_410m(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/pythia-410m-deduped", quantization=quantization)


# checked
class pythia_1b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/pythia-1b-deduped", quantization=quantization)
        

# checked
class pythia_1_4b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/pythia-1.4b-deduped", quantization=quantization)


# checked
class pythia_2_8b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/pythia-2.8b-deduped", quantization=quantization)
        

# checked
class pythia_6_9b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/pythia-6.9b-deduped", quantization=quantization)
        

# checked
class pythia_12b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/pythia-12b-deduped", quantization=quantization)
        

# checked
class gpt_neo_125m(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/gpt-neo-125M", quantization=quantization)
        

# checked
class gpt_neo_1_3b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/gpt-neo-1.3B", quantization=quantization)
        

# checked
class gpt_neo_2_7b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/gpt-neo-2.7B", quantization=quantization)
        

# checked
class gpt_neox_20b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/gpt-neox-20b", quantization=quantization)
        

# checked
class gpt_j_6b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="EleutherAI/gpt-j-6B", quantization=quantization)
    

# checked
class opt_125m(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="facebook/opt-125m", quantization=quantization)
        
    
# checked
class opt_350m(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="facebook/opt-350m", quantization=quantization)
        

# checked
class opt_1_3b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="facebook/opt-1.3B", quantization=quantization)
        
    
# checked
class opt_2_7b(GPT2Model):
    def __init__(self, quantization):
        super().__init__(model_name="facebook/opt-2.7b", quantization=quantization)
        
    
    
    
    
    
class LlamaModel:
    def __init__(self, model_name, quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cahce_dir=HF_CACHE_DIR_NAME,
            device_map = 'auto'
        )
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                quantization_config=bnb_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto'
            )
        try:
            if self.model.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        except:
            if self.model.config.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()
        # print(prompt_text)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        # for k, v in inputs.items():
        #     inputs[k] = v.to(self.model.device)
        outputs = self.model(**inputs)
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}
        # outputs.logits = outputs.logits.to('cpu')
        # logits = outputs.logits[0, -1]
        logits = outputs['logits'][0, -1]
        print(logits.device)
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
    def process_question_poe(self, question, strat="below_average"):
        prompt_text_elimination = question.get_natural_prompt()
        inputs = self.tokenizer(prompt_text_elimination, return_tensors="pt", return_token_type_ids=False)
        # for k, v in inputs.items():
        #     inputs[k] = v.to(self.model.device)
        outputs = self.model(**inputs)
        inputs = {k: v.cpu().detach() for k, v in inputs.items()}
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}
        # for k, v in outputs.items():
        #     outputs[k] = v.cpu().detach()
        #     print(f"{k}: {v.device} ~~")
        # logits = outputs.logits[0, -1]
        logits = outputs['logits'][0, -1]
        probs = logits.float().softmax(dim=-1)
        
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # print(logprobs_dict)
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            ) # [:50]
        }
        
        choices_char = [idx_to_ltr(i) for i in range(question.get_n_choices())]
        token_prefix = distinguish_prefix(logprobs_dict, choices_char)
        
        eliminiation_prob_dict = dict()
        for char in choices_char:
            eliminiation_prob_dict[char] = logprobs_dict[f"{token_prefix}{char}"]
                    
        if strat == "below_average":
            average_prob = 0.0
            for char in choices_char:
                average_prob += eliminiation_prob_dict[char]
            average_prob /= len(choices_char)
            
            for char in choices_char:
                if eliminiation_prob_dict[char] < average_prob:
                    eliminiation_prob_dict[char] = -100.0
        elif strat == "min":
            for i, char in enumerate(choices_char):
                if i == 0:
                    min_prob = eliminiation_prob_dict[char]
                else:
                    if eliminiation_prob_dict[char] < min_prob:
                        min_prob = eliminiation_prob_dict[char]

            for char in choices_char:
                if eliminiation_prob_dict[char] == min_prob:
                    eliminiation_prob_dict[char] = -100.0
        
        prompt_text_prediction = question.get_poe_prompt(eliminiation_prob_dict)
        inputs = self.tokenizer(prompt_text_prediction, return_tensors="pt", return_token_type_ids=False)
        
        outputs = self.model(**inputs)
        inputs = {k: v.cpu().detach() for k, v in inputs.items()}
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}
        # outputs.logits = outputs.logits.to('cpu')
        # logits = outputs.logits[0, -1]
        logits = outputs['logits'][0, -1]
        probs = logits.float().softmax(dim=-1)
        
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }             
 
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }
        logprobs_dict = mask_logprobs(token_prefix, logprobs_dict, eliminiation_prob_dict)
        
        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 3:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 3:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0][-(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )

    def process_question_cot(self, question):
        prompt_text = question.get_cot_prompt()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'])
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is "
        
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        for k, v in second_inputs.items():
            second_inputs[k] = v.to(self.model.device)
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
        
    def process_question_ps(self, question):
        prompt_text = question.get_ps_prompt()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'])
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is "
        
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        for k, v in second_inputs.items():
            second_inputs[k] = v.to(self.model.device)
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        

# checked
class Mistral(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="mistralai/Mistral-7B-v0.1", quantization=quantization)
    

# checked
class Mixtral(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="mistralai/Mixtral-8x7B-v0.1", quantization=quantization)
        

# checked
class Mistral_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="mistralai/Mistral-7B-Instruct-v0.2", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Mixtral_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Llama_2_7B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Llama-2-7b-hf", quantization=quantization)
        

# checked
class Llama_2_13B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Llama-2-13b-hf", quantization=quantization)
        

# checked
class Llama_2_70B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Llama-2-70b-hf", quantization=quantization)
        

# checked
class Llama_2_7B_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Llama-2-7b-chat-hf", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class Llama_2_13B_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Llama-2-13b-chat-hf", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Llama_2_70B_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Llama-2-70b-chat-hf", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    
    
class Llama_3_8B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Meta-Llama-3-8B", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    
    
class Llama_3_8B_Instruct(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Meta-Llama-3-8B-Instruct", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    
    
class Llama_3_70B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Meta-Llama-3-70B", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    
    
class Llama_3_70B_Instruct(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="meta-llama/Meta-Llama-3-70B-Instruct", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    
    
# checked
class Vicuna_1_7B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="lmsys/vicuna-7b-v1.3", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Vicuna_1_13B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="lmsys/vicuna-13b-v1.3", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class Vicuna_1_33B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="lmsys/vicuna-33b-v1.3", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Vicuna_2_7B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="lmsys/vicuna-7b-v1.5-16k", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class Vicuna_2_13B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="lmsys/vicuna-13b-v1.5-16k", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Orca_2_7B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="microsoft/Orca-2-7b", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Orca_2_13B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="microsoft/Orca-2-13b", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class OpenChat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="openchat/openchat_8192", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class OpenChat_2(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="openchat/openchat_v2", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class OpenChat_2_W(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="openchat/openchat_v2_w", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class OpenChat_3_2(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="openchat/openchat_v3.2", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class OpenChat_3_2_Super(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="openchat/openchat_v3.2_super", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class OpenChat_3_5(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="openchat/openchat-3.5-0106", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Starling(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="berkeley-nest/Starling-LM-7B-alpha", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Zephyr_Alpha(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="HuggingFaceH4/zephyr-7b-alpha", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Zephyr_Beta(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="HuggingFaceH4/zephyr-7b-beta", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Upstage_Llama_1_30B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="upstage/llama-30b-instruct-2048", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Upstage_Llama_1_65B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="upstage/llama-65b-instruct", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Upstage_Llama_2_70B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="upstage/Llama-2-70b-instruct", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class Solar_70B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="upstage/SOLAR-0-70b-16bit", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class Solar_10_7B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="upstage/SOLAR-10.7B-v1.0", quantization=quantization)
        

# checked
class Solar_10_7B_Instruct(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="upstage/SOLAR-10.7B-instruct-v1.0", quantization=quantization)
        

# checked
class Yi_6B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="01-ai/Yi-6B-200K", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            print(f"answer_tokens: {self.tokenizer.decode(answer_tokens)}")
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )


# checked
class Yi_34B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="01-ai/Yi-34B-200K", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            print(f"answer_tokens: {self.tokenizer.decode(answer_tokens)}")
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
        

# checked
class Yi_6B_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="01-ai/Yi-6B-Chat", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Yi_34B_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="01-ai/Yi-34B-Chat", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
    # Template may need revision to fit the chat format. Leaving it as-is for now.


# checked
class Amber(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="LLM360/Amber", quantization=quantization)
    

# checked
class Amber_Chat(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="LLM360/AmberChat", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Crystal_Coder(LlamaModel):
    def __init__(self, quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalCoder",
            cahce_dir=HF_CACHE_DIR_NAME,
            device_map = 'auto', 
            trust_remote_code=True
        )
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "LLM360/CrystalCoder",
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                quantization_config=bnb_config, 
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "LLM360/CrystalCoder",
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                trust_remote_code=True
            )
        try:
            if self.model.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        except:
            if self.model.config.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            print(f"answer_tokens: {self.tokenizer.decode(answer_tokens)}")
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
        

# checked
class Crystal_Chat(LlamaModel):
    def __init__(self, quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "LLM360/CrystalChat",
            cahce_dir=HF_CACHE_DIR_NAME,
            device_map = 'auto', 
            trust_remote_code=True
        )
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "LLM360/CrystalChat",
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                quantization_config=bnb_config, 
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "LLM360/CrystalChat",
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                trust_remote_code=True
            )
        try:
            if self.model.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        except:
            if self.model.config.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Falcon_7B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="tiiuae/falcon-7b", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            print(f"answer_tokens: {self.tokenizer.decode(answer_tokens)}")
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
        

# checked
class Falcon_40B(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="tiiuae/falcon-40b", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
        

# checked
class Falcon_7B_Instruct(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="tiiuae/falcon-7b-instruct", quantization=quantization)
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Falcon_40B_Instruct(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="tiiuae/falcon-40b-instruct", quantization=quantization)
    
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 2:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 2:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, -(choice_n_tokens):]
            print(f"answer_tokens: {self.tokenizer.decode(answer_tokens)}")
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    

# checked
class Solar_OrcaDPO_Solar_Instruct_Slerp(LlamaModel):
    def __init__(self, quantization):
        super().__init__(model_name="kodonho/Solar-OrcaDPO-Solar-Instruct-SLERP", quantization=quantization)
    # Template may need revision to fit the chat format. Leaving it as-is for now.
    








class T5Model:
    def __init__(self, model_name, quantization=False):
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cahce_dir=HF_CACHE_DIR_NAME,
            device_map = 'auto'
        )
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto', 
                quantization_config=bnb_config
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR_NAME,
                use_cache=False,
                device_map = 'auto'
            )
        try:
            if self.model.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        except:
            if self.model.config.vocab_size != len(self.tokenizer):
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        decoder_inputs = self.tokenizer("", return_tensors="pt", return_token_type_ids=False).input_ids
        outputs = self.model(**inputs, decoder_input_ids=decoder_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 3:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            decoder_inputs = self.tokenizer("", return_tensors="pt", return_token_type_ids=False).input_ids
            outputs = self.model(**inputs, decoder_input_ids=decoder_inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 3:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0][-(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            decoder_inputs = self.tokenizer("", return_tensors="pt", return_token_type_ids=False).input_ids
            outputs = self.model(**inputs, decoder_input_ids=decoder_inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )


class flan_t5_small(T5Model):
    def __init__(self, quantization=False):
        super().__init__(model_name="google/flan-t5-small", quantization=quantization)


class flan_t5_base(T5Model):
    def __init__(self, quantization=False):
        super().__init__(model_name="google/flan-t5-base", quantization=quantization)
        

class flan_t5_large(T5Model):
    def __init__(self, quantization=False):
        super().__init__(model_name="google/flan-t5-large", quantization=quantization)


class flan_t5_xl(T5Model):
    def __init__(self, quantization=False):
        super().__init__(model_name="google/flan-t5-xl", quantization=quantization)
        

class flan_t5_xxl(T5Model):
    def __init__(self, quantization=False):
        super().__init__(model_name="google/flan-t5-xxl", quantization=quantization)






class OpenAIModel:
    def __init__(self, api_key, model_name, add_space=False):
        self.client = openai.OpenAI(api_key=api_key)
        self.add_space = add_space
        self.model_name = model_name

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()
        response = self._get_response(text=prompt_text)
        logprobs_raw = response.choices[0].logprobs.content[0].top_logprobs
        logprobs = dict()
        for logprob_object in logprobs_raw:
            logprobs[logprob_object.token] = logprob_object.logprob

        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=[]
        )

    def _get_response(self, text):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text+(" " if self.add_space else "")}
                    ],
                    temperature=0,  # Doesn't actually matter here
                    max_tokens=1,  # Just need to get letter
                    logprobs=True,
                    top_logprobs=5, # Get top 5 logprobs
                )
                return response
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)
                
    def _get_rationale_response(self, text):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text+(" " if self.add_space else "")}
                    ],
                    temperature=0.5,  # Doesn't actually matter here
                    max_tokens=512,  # Just need to get letter
                    # logprobs=True,
                    # top_logprobs=5, # Get top 5 logprobs
                )
                return str(response.choices[0].message.content)
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)

    def process_question_brown(self, question):
        print("I need to check gpt3.5 or later enable us to use 'echo' parameter")
        
        raise NotImplementedError
    
    def process_question_poe(self, question):
        print("I need to check gpt3.5 or later enable us to use 'echo' parameter")
        
        raise NotImplementedError

    def process_question_cot(self, question):
        prompt_text = question.get_cot_prompt()
        cot_rationale = self._get_rationale_response(text=prompt_text)
        
        second_text = cot_rationale + ". Therefore, the answer is "
        response = self._get_response(text=second_text)
        
        logprobs_raw = response.choices[0].logprobs.content[0].top_logprobs
        logprobs = dict()
        for logprob_object in logprobs_raw:
            logprobs[logprob_object.token] = logprob_object.logprob

        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=[]
        )
        
    def process_question_ps(self, question):
        prompt_text = question.get_ps_prompt()
        cot_rationale = self._get_rationale_response(text=prompt_text)
        
        second_text = cot_rationale + ". Therefore, the answer is "
        response = self._get_response(text=second_text)
        
        logprobs_raw = response.choices[0].logprobs.content[0].top_logprobs
        logprobs = dict()
        for logprob_object in logprobs_raw:
            logprobs[logprob_object.token] = logprob_object.logprob

        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=[]
        )

class GPT3_BABBAGE(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT3_BABBAGE_MODEL_NAME)
        
    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()
        response = self._get_response(text=prompt_text, echo=False)
        logprobs = response.choices[0].logprobs.top_logprobs[0]

        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=[]
        )
        
        
    def _get_response(self, text, echo):
        while True:
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=text+(" " if self.add_space else ""),
                    temperature=0,  # Doesn't actually matter here
                    max_tokens=1,  # Just need to get letter
                    logprobs=5, # Get top 5 logprobs
                    echo=echo
                )
                return response
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)
                
                
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()

        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()

        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)

            # Get unconditional logprobs
            response = self._get_response(text=f"Answer: {choice}", echo=True)
            choice_logprobs = response.choices[0].logprobs.token_logprobs[2:-1]

            choice_n_tokens = len(choice_logprobs)
            unconditional_logprobs[ltr] = sum(choice_logprobs)
            lens[ltr] = choice_n_tokens

            # Get conditional logprobs
            response = self._get_response(
                text=f"{prompt_text} {choice}", echo=True
            )
            token_logprobs = (
                response.choices[0].logprobs.token_logprobs
            )
            choice_logprobs = token_logprobs[-(choice_n_tokens+1):-1]

            logprobs[ltr] = sum(choice_logprobs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )


class GPT3_DAVINCI(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT3_DAVINCI_MODEL_NAME)

    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt()
        response = self._get_response(text=prompt_text, echo=False)
        logprobs = response.choices[0].logprobs.top_logprobs[0]

        return ModelResponseNatural(
            logprobs=logprobs,
            response_list=[]
        )
        
    def _get_response(self, text, echo):
        while True:
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=text+(" " if self.add_space else ""),
                    temperature=0,  # Doesn't actually matter here
                    max_tokens=1,  # Just need to get letter
                    logprobs=5, # Get top 5 logprobs
                    echo=echo
                )
                return response
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)
                  
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()

        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()

        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)

            # Get unconditional logprobs
            response = self._get_response(text=f"Answer: {choice}", echo=True)
            choice_logprobs = response.choices[0].logprobs.token_logprobs[2:-1]

            choice_n_tokens = len(choice_logprobs)
            unconditional_logprobs[ltr] = sum(choice_logprobs)
            lens[ltr] = choice_n_tokens

            # Get conditional logprobs
            response = self._get_response(
                text=f"{prompt_text} {choice}", echo=True
            )
            token_logprobs = (
                response.choices[0].logprobs.token_logprobs
            )
            choice_logprobs = token_logprobs[-(choice_n_tokens+1):-1]

            logprobs[ltr] = sum(choice_logprobs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )


class GPT3_5(OpenAIModel):

    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT3_5_MODEL_NAME)
        
        
class GPT4(OpenAIModel):
    
    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT4_MODEL_NAME)
        
        
class GPT4o(OpenAIModel):
    
    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=GPT4O_MODEL_NAME)
        
        
class AnthropicModel:
    def __init__(self, api_key, model_name, add_space=False):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.add_space = add_space
        self.model_name = model_name

    def _get_response(self, text):
        while True:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text+(" " if self.add_space else "")}
                    ],
                    temperature=0.0,  # Doesn't actually matter here
                    max_tokens=1
                )
                return response
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)
                
    def _get_rationale_response(self, text):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text+(" " if self.add_space else "")}
                    ],
                    temperature=0.5,
                    max_tokens=512,  # Just need to get letter
                )
                return str(response.content[0].text)
            except Exception as e:
                print(e)
                print("Will wait and retry...")
                time.sleep(RETRY_SLEEP_TIME)
                
    def process_question_natural(self, question):
        prompt_text = question.get_natural_prompt(chat=True)
        response = self._get_response(text=prompt_text + " ")
        response_text = response.content[0].text
        print(response_text)

        return ModelResponseNatural(
            logprobs=None,
            response_list=[response_text]
        )

    def process_question_brown(self, question):
        print("I need to check gpt3.5 or later enable us to use 'echo' parameter")
        
        raise NotImplementedError
    
    def process_question_poe(self, question):
        print("I need to check gpt3.5 or later enable us to use 'echo' parameter")
        
        raise NotImplementedError

    def process_question_cot(self, question):
        prompt_text = question.get_cot_prompt()
        cot_rationale = self._get_rationale_response(text=prompt_text)
        
        second_text = cot_rationale + "Let's response in a single uppercase alphabet."
        response = self._get_response(text=second_text)
        response_text = response.content[0].text

        return ModelResponseNatural(
            logprobs=None,
            response_list=[response_text]
        )
        
    def process_question_ps(self, question):
        prompt_text = question.get_ps_prompt()
        cot_rationale = self._get_rationale_response(text=prompt_text)
        
        second_text = cot_rationale + "Let's response in a single uppercase alphabet."
        response = self._get_response(text=second_text)
        response_text = response.content[0].text

        return ModelResponseNatural(
            logprobs=None,
            response_list=[response_text]
        )
        
class Claude3_Haiku(AnthropicModel):
    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=CLAUDE3_HAIKU_MODEL_NAME)
        
        
class Claude3_Sonnet(AnthropicModel):
    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=CLAUDE3_SONNET_MODEL_NAME)
        
        
class Claude3_Opus(AnthropicModel):
    def __init__(self, api_key):
        super().__init__(api_key=api_key, model_name=CLAUDE3_OPUS_MODEL_NAME)
        
        
        
def get_model_by_name(name, api_key, quantization):
    try:
        return {
            "gpt3_babbage": GPT3_BABBAGE,
            "gpt3_davinci": GPT3_DAVINCI,
            "gpt3.5": GPT3_5,
            "gpt4": GPT4,
            "gpt4o": GPT4o,
            "claude3_haiku": Claude3_Haiku,
            "claude3_sonnet": Claude3_Sonnet,
            "claude3_opus": Claude3_Opus,
        }[name](api_key=api_key)
    except KeyError:
        return {
            "mistral": Mistral,
            "mixtral": Mixtral,
            "mistral_chat": Mistral_Chat,
            "mixtral_chat": Mixtral_Chat,
            "llama_2_7b": Llama_2_7B,
            "llama_2_13b": Llama_2_13B,
            "llama_2_70b": Llama_2_70B,
            "llama_2_7b_chat": Llama_2_7B_Chat,
            "llama_2_13b_chat": Llama_2_13B_Chat,
            "llama_2_70b_chat": Llama_2_70B_Chat,
            "llama_3_8b": Llama_3_8B,
            "llama_3_8b_instruct": Llama_3_8B_Instruct,
            "llama_3_70b": Llama_3_70B,
            "llama_3_70b_instruct": Llama_3_70B_Instruct,
            "vicuna_1_7b": Vicuna_1_7B, # lmsys/vicuna-7b-v1.3
            "vicuna_1_13b": Vicuna_1_13B, # lmsys/vicuna-13b-v1.3
            "vicuna_1_33b": Vicuna_1_33B, # lmsys/vicuna-33b-v1.3
            "vicuna_2_7b": Vicuna_2_7B, # lmsys/vicuna-7b-v1.5-16k
            "vicuna_2_13b": Vicuna_2_13B, # lmsys/vicuna-13b-v1.5-16k
            "orca_2_7b": Orca_2_7B, # microsoft/Orca-2-7b
            "orca_2_13b": Orca_2_13B, # microsoft/Orca-2-13b
            "openchat": OpenChat, # openchat/openchat_8192
            "openchat_2": OpenChat_2, # openchat/openchat_v2
            "openchat_2_w": OpenChat_2_W, # openchat/openchat_v2_w
            "openchat_3.2": OpenChat_3_2, # openchat/openchat_v3.2
            "openchat_3.2_super": OpenChat_3_2_Super, # openchat/openchat_v3.2_super
            "openchat_3.5": OpenChat_3_5, # openchat/openchat-3.5-0106
            "starling": Starling, # berkeley-nest/Starling-LM-7B-alpha
            "zephyr_alpha": Zephyr_Alpha, # HuggingFaceH4/zephyr-7b-alpha
            "zephyr_beta": Zephyr_Beta, # HuggingFaceH4/zephyr-7b-beta
            "upstage_llama_1_30b": Upstage_Llama_1_30B, # upstage/llama-30b-instruct-2048
            "upstage_llama_1_65b": Upstage_Llama_1_65B, # upstage/llama-65b-instruct
            "upstage_llama_2_70b": Upstage_Llama_2_70B, # upstage/Llama-2-70b-instruct
            "solar_70b": Solar_70B, # upstage/SOLAR-0-70b-16bit
            "solar_10.7b": Solar_10_7B, # upstage/SOLAR-10.7B-v1.0
            "solar_10.7b_instruct": Solar_10_7B_Instruct, # upstage/SOLAR-10.7B-instruct-v1.0
            "yi_6b": Yi_6B, # 01-ai/Yi-6B-200K
            "yi_34b": Yi_34B, # 01-ai/Yi-34B-200K
            "yi_6b_chat": Yi_6B_Chat, # 01-ai/Yi-6B-Chat
            "yi_34b_chat": Yi_34B_Chat, # 01-ai/Yi-34B-Chat
            "amber": Amber, # LLM360/Amber
            "amber_chat": Amber_Chat, # LLM360/AmberChat
            "crystal_coder": Crystal_Coder, # LLM360/CrystalCoder
            "crystal_chat": Crystal_Chat, # LLM360/CrystalChat
            "falcon_7b": Falcon_7B, # tiiuae/falcon-7b
            "falcon_40b": Falcon_40B, # tiiuae/falcon-40b
            "falcon_7b_instruct": Falcon_7B_Instruct, # tiiuae/falcon-7b-instruct
            "falcon_40b_instruct": Falcon_40B_Instruct, # tiiuae/falcon-40b-instruct
            "solar_orcadpo_solar_instruct_slerp": Solar_OrcaDPO_Solar_Instruct_Slerp, # kodonho/Solar-OrcaDPO-Solar-Instruct-SLERP

            # "pythia_70m": pythia_70m, # EleutherAI/pythia-70m-deduped
            # "pythia_160m": pythia_160m, # EleutherAI/pythia-160m-deduped
            "pythia_410m": pythia_410m, # EleutherAI/pythia-410m-deduped
            "pythia_1b": pythia_1b, # EleutherAI/pythia-1b-deduped
            "pythia_1.4b": pythia_1_4b, # EleutherAI/pythia-1.4b-deduped
            "pythia_2.8b": pythia_2_8b, # EleutherAI/pythia-2.8b-deduped
            "pythia_6.9b": pythia_6_9b, # EleutherAI/pythia-6.9b-deduped
            "pythia_12b": pythia_12b, # EleutherAI/pythia-12b-deduped
            "gpt_neo_125m": gpt_neo_125m,
            "gpt_neo_1.3b": gpt_neo_1_3b, # EleutherAI/gpt-neo-1.3b
            "gpt_neo_2.7b": gpt_neo_2_7b, # EleutherAI/gpt-neo-2.7b
            "gpt_neox_20b": gpt_neox_20b, # EleutherAI/gpt-neox-20b
            "gpt_j_6b": gpt_j_6b, # EleutherAI/gpt-j-6b
            "opt_125m": opt_125m, # facebook/opt-125m
            "opt_350m": opt_350m, # facebook/opt-350m
            "opt_1.3b": opt_1_3b, # facebook/opt-1.3b
            "opt_2.7b": opt_2_7b, # facebook/opt-2.7b
            
            "flan_t5_small": flan_t5_small, # google/flan-t5-small
            "flan_t5_base": flan_t5_base, # google/flan-t5-base
            "flan_t5_large": flan_t5_large, # google/flan-t5-large
            "flan_t5_xl": flan_t5_xl, # google/flan-t5-xl
            "flan_t5_xxl": flan_t5_xxl, # google/flan-t5-xxl
        }[name](quantization=quantization)
