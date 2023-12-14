from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Dummy():
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 format_instruction=None
                 ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.format_instruction = format_instruction

    def generate(self, inp):
        return inp

    def tokenizer(
            self, 
            instr, 
            padding=None,
            truncation=None,
            return_tensors=None,
            max_length=None,
            ):
        
        return {
            'instruction': instr,
            }