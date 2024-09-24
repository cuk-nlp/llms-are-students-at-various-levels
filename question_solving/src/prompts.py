import random
from dataclasses import dataclass

from utils.base_utils import idx_to_ltr


@dataclass
class QuestionPart:
    text: str
    tag: str = None

    def __str__(self):
        if self.tag is not None:
            return f"{self.tag}: {self.text}"
        else:
            return self.text


@dataclass
class Question:
    question_id: int
    parts: list
    choices: list
    answer_idx: int
    hint: str
    use_hint: bool

    def get_n_choices(self):
        return len(self.choices)
    
    def get_answer_str(self):
        return self.choices[self.answer_idx]
    
    def _get_prompt(self, include_choices, chat):
        prompt = ""
        for part in self.parts:
            prompt += f"{str(part)}\n"
        if include_choices:
            for i, choice in enumerate(self.choices):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        if self.use_hint:
            prompt += f"\nHints: \n{self.hint}\n\n"
        if chat:
            return prompt + "Let's response in a single uppercase alphabet."
        else:
            return prompt + "Answer: "

    def _get_poe_prompt(self, mask_dict):
        prompt = ""
        for part in self.parts:
            prompt += f"{str(part)}\n"
        for i, choice in enumerate(self.choices):
            if mask_dict[idx_to_ltr(i)] == -100.0:
                prompt += f"{idx_to_ltr(i)}. [MASK]\n"
            else:
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        if self.use_hint:
            prompt += f"\nHints: \n{self.hint}\n\n"
        return prompt + "Answer: "

    def _get_cot_prompt(self, include_choices):
        prompt = ""
        for part in self.parts:
            prompt += f"{str(part)}\n"
        if include_choices:
            for i, choice in enumerate(self.choices):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        if self.use_hint:
            prompt += f"\nHints: \n{self.hint}\n\n"
        return prompt + "Let's think step by step. "
    
    def _get_ps_prompt(self, include_choices):
        prompt = ""
        for part in self.parts:
            prompt += f"{str(part)}\n"
        if include_choices:
            for i, choice in enumerate(self.choices):
                prompt += f"{idx_to_ltr(i)}. {choice}\n"
        if self.use_hint:
            prompt += f"\nHints: \n{self.hint}\n\n"
        return prompt + "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step. "
    
    def get_natural_prompt(self, chat=False):
        return self._get_prompt(include_choices=True, chat=chat)
    
    def get_brown_prompt(self):
        return self._get_prompt(include_choices=False)
    
    def get_poe_prompt(self, mask_dict):
        return self._get_poe_prompt(mask_dict)
    
    def get_cot_prompt(self):
        return self._get_cot_prompt(include_choices=True)
    
    def get_ps_prompt(self):
        return self._get_ps_prompt(include_choices=True)
    
    def permute_choices(self, perm):
        self.choices = [self.choices[i] for i in perm]
        self.answer_idx = perm.index(self.answer_idx)
        

class QuestionWithExemplar(Question):
    def __init__(self, parts, choices, answer_idx, hint, exemplar, base_prompt, use_hint, question_id):
        super().__init__(question_id, parts, choices, answer_idx, hint, use_hint)
        self.exemplar = exemplar
        self.base_prompt = base_prompt
    
    def get_natural_prompt(self, chat=False):
        prompt = super().get_natural_prompt(chat=chat)
        if len(self.exemplar):
            examplar_prompts = [e.get_natural_prompt() for e in self.exemplar]
            exemplar = "\n\n".join(examplar_prompts)
            return f"{self.base_prompt}\n\n{exemplar}\n\n{prompt}"
        else:
            return f"{self.base_prompt}\n\n{prompt}"
        
    def get_brown_prompt(self):
        prompt = super().get_brown_prompt()
        if len(self.exemplar):
            exemplar_prompts = [e.get_brown_prompt() for e in self.exemplar]
            exemplar = "\n\n".join(exemplar_prompts)
            return f"{self.base_prompt}\n\n{exemplar}\n\n{prompt}"
        else:
            return f"{self.base_prompt}\n\n{prompt}"
        
    
class Exemplar(Question):
    def get_natural_prompt(self):
        prompt = super().get_natural_prompt()
        answer_ltr = idx_to_ltr(self.answer_idx)
        return f"{prompt} {answer_ltr}"
    
    def get_brown_prompt(self):
        prompt = super().get_brown_prompt()
        return f"{prompt} {self.get_answer_str()}"
