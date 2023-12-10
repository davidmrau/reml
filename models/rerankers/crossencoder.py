from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch

from transformers import DefaultDataCollator, AutoModel, AutoTokenizer
class CrossEncoder:
    def __init__(self, model_name=None):

        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def __call__(self, kwargs):
        score = self.model(**kwargs.to(self.device)).logits
        return {
                "score": score
            }
        
    def collate_fn(self, examples):
        question = [e['query'] for e in examples]
        doc= [e['doc'] for e in examples]
        q_id = [e['q_id'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        inp_dict = self.tokenizer(question, doc, padding=True, truncation='only_second', return_tensors='pt')
        inp_dict['q_id'] = q_id
        inp_dict['d_id'] = d_id
        return inp_dict

