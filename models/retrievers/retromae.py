from transformers import AutoModel, AutoTokenizer
import torch

class RetroMAE:
    def __init__(self, model_name=None):

        self.model_name = "Shitao/RetroMAE"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.model = self.model.to(self.device)
        self.model.eval()

        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

    def collate_fn(self, batch, query_or_doc=None):
        content = [sample['content'] for sample in batch]
        return_dict = self.tokenizer(content, padding=True, truncation=True, return_tensors='pt')
        return return_dict

    def __call__(self, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = outputs[0][:,0]
        return {
                "embedding": emb
            }

