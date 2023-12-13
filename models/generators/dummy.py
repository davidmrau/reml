from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Dummy():
    def __init__(self, model_name=None, max_new_tokens=1, format_instruction=None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.format_instruction = format_instruction

    def generate(self, inp):
        return inp

    def tokenize(self, example):
        inp_dict = defaultdict()
        inp_dict['id'] = example["id"]
        inp_dict['content'] = example["content"]
        return inp_dict

    def collate_fn(self, examples):
        ids = [e['q_id'] for e in examples]
        inp_dict = dict()
        instr= [self.format_instruction(e) for e in examples]
        inp_dict['q_id'] = ids
        inp_dict['instruction'] = instr
        inp_dict['inp'] = instr
        return inp_dict
