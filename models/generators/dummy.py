from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Dummy():
    def __init__(self, kwargs):
        self.model_name = kwargs['model_name']
        #self.top_k_documents = kwargs['top_k_documents']
        if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
            raise NotImplemtedError('Only batch size 1 is implemented yet.')
        self.batch_size = kwargs['batch_size']
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, query, documents):
        instrucion = self.format_instruction(query, documents)
        generated_response = instrucion
        return instrucion, generated_response

    def format_instruction(self, query, docs):
        instruction_prompt = '### Instruction: Please give answer given query and support documents'
        docs_prompt = ''
        for i, doc in enumerate(docs['sentence']):
            docs_prompt += f"### Documents {i}: {doc} "
        query_prompt = f'### Query: {query}'
        response_prompt = '### Response:'
        return f"""{instruction_prompt}
{query_prompt}
{docs_prompt}
{response_prompt}"""


    def tokenize(self, example, sentence_field, id_field):
        inp_dict = defaultdict()
        # inp_dict = self.tokenizer(example[sentence_field], truncation=True)
        inp_dict['id'] = example[id_field]
        inp_dict['sentence'] = example[sentence_field]
        return inp_dict
