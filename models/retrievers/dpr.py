from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

class DPR:
    def __init__(self, kwargs):

        self.model_name = kwargs['model_name']
        self.model = AutoModel.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.device = torch.device('cpu') #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithId(tokenizer=self.tokenizer)

    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    @torch.no_grad()
    def encode(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.data_collator)
        embs_list = list()
        ids = list()
        for batch in dataloader:
            batch_ids = batch.pop('ids')
            ids += batch_ids
            outputs = self.model(**batch.to(self.device))
            # pooling over hidden representations
            emb = self.mean_pooling(outputs[0], batch['attention_mask'])
            embs_list.append(emb)

        embs = torch.cat(embs_list)

        return {
            "ids": ids,
            "embeddings": embs
            }

    def index(self, dataloader, batch_size):
        pass

    def tokenize(self, example, sentence_field, id_field):
        inp_dict = self.tokenizer(example[sentence_field], truncation=True)
        inp_dict['id'] = example[id_field]
        return inp_dict

# def collate_fn(self, examples):
#     doc = [e['claim'] for e in examples]
#     input_tokenized = self.tokenizer(question, padding=True, truncation='only_second', max_length=self.tokenizer.max_length, return_tensors='pt')
#     return {
#     **input_tokenized
#     }
