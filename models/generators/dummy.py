from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Dummy():
    def __init__(self, model_name=None, max_new_tokens=1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def generate(self, inp):
        return inp

    def tokenize(self, example):
        inp_dict = defaultdict()
        inp_dict['id'] = example["id"]
        inp_dict['sentence'] = example["sentence"]
        return inp_dict
