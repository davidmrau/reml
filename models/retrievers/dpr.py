from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

class DPR:
    def __init__(self, model_name=None):

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithId(tokenizer=self.tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def __call__(self, kwargs):
        outputs = self.model(**kwargs.to(self.device))
        # pooling over hidden representations
        emb = self.mean_pooling(outputs[0], kwargs['attention_mask'])
        return {
                "embedding": emb
            }
        
    def tokenize(self, example):
        inp_dict = self.tokenizer(example["sentence"], truncation=True)
        inp_dict['id'] = example["id"]
        return inp_dict
