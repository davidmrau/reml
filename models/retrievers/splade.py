from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

class Splade:
    def __init__(self, model_name):

        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.collate_fn = DataCollatorWithId(tokenizer=self.tokenizer)


    def __call__(self, kwargs):
        kwargs = {key: value.to('cuda') for key, value in kwargs.items()}
        outputs = self.model(**kwargs.to(self.device))
        # pooling over hidden representations
        emb, _ = torch.max(torch.log(1 + torch.relu(outputs)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
        return {
                "embedding": emb
            }

    def tokenize(self, example):
        inp_dict = self.tokenizer(example["content"], truncation=True)
        return inp_dict
