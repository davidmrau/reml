from transformers import AutoModel, AutoTokenizer, DefaultDataCollator
from torch.utils.data import DataLoader
import torch
class DPR:
    def __init__(self, model_name=None):

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.collate_fn = DefaultDataCollator()
    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def __call__(self, batch):
        batch = {key: value.to('cuda') for key, value in batch.items()}
        outputs = self.model(**batch)
        # pooling over hidden representations
        emb = self.mean_pooling(outputs[0], batch['attention_mask'])
        return {
                "embedding": emb
            }
        
    def tokenize(self, example):
        inp =  self.tokenizer(example["sentence"], truncation=True, padding='max_length')
        return inp
