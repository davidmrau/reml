from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Dummy():
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1,
                 ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens


    def gold_answer(self, labels):
        return [label[0] for label in labels]

    def tokenizer(
            self, 
            instr, 
            padding=None,
            truncation=None,
            return_tensors=None,
            max_length=None,
            ):
        return instr



    def get_response_marker(self):
        return " Response: "

    def format_instruction(self, sample):
        if 'doc' in sample:
            docs_prompt = ''
            for i, doc in enumerate(sample['doc']):
                docs_prompt += f"Document {i+1}: {doc}\n"
            return f"""Please answer the question given the support documents:\n Question: {sample['query']}\n Support Documents:\n{docs_prompt}\n{self.get_response_marker()} """
        else:
            return f"""Please answer the question.\n Question: {sample['query']}\n{self.get_response_marker()}"""
    
    def generate(self, inp):
        cleaned_generated_response = [gen.split(self.get_response_marker(), 1)[1] for gen in inp]
        return cleaned_generated_response