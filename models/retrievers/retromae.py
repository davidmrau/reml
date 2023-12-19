from transformers import AutoModel, AutoTokenizer, DefaultDataCollator
import torch
class RetroMAE:
    def __init__(self, model_name=None):

        self.model_name = "Shitao/RetroMAE"
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.model)
        print(self.device)
        self.model = self.model.to(self.device)
        self.collate_fn = DefaultDataCollator()

    def cls_pooling(self, model_output, attention_mask):
        return model_output[:,0]

    def __call__(self, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = self.cls_pooling(outputs[0])
        return {
                "embedding": emb
            }

    def tokenize(self, example):
        inp =  self.tokenizer(example["content"], truncation=True, padding='max_length')
        return inp
