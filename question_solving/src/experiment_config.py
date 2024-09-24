from dataclasses import dataclass
from config.constants import QUESTION_SOLVING_RESULTS_DIR_NAME

@dataclass
class ExperimentConfig:
    ds_name: str
    prompt_style: str
    model_name: str
    style_name: str
    n_shots: int
    do_perm: bool
    quantization: bool
    
    def get_save_fname(self):
        vals = [str(v) for v in vars(self).values()]
        return f"{QUESTION_SOLVING_RESULTS_DIR_NAME}/{'_'.join(vals)}.pkl"