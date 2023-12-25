from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel, PeftConfig



class RepLlama:
    def __init__(self, model_name=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='right')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.get_model(model_name)
        self.model.max_length = 256


    def get_model(self, peft_model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map='auto', torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        model.eval()
        return model

    def collate_fn(self, batch, query_or_doc):
        if query_or_doc == 'doc':
            content = [f"query: {sample['content']}{self.tokenizer.eos_token}" for sample in batch]
        else:
            content = [f"passage: {sample['content']}{self.tokenizer.eos_token}" for sample in batch]

        return_dict = self.tokenizer(content, padding=True, truncation=True, return_tensors='pt')
        return return_dict

    def __call__(self, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = outputs[0]
        first_eos_indices = ((kwargs['input_ids'] == self.tokenizer.pad_token_id).cumsum(dim=1) == 1).nonzero()[:, 1]
        # Gather the embeddings based on the first EOS indices
        first_eos_emb = emb[torch.arange(emb.size(0)), first_eos_indices]
        #first_eos_emb = emb[:, -1]
        normed_emb = torch.nn.functional.normalize(first_eos_emb, p=2, dim=0)

        return {
                "embedding": normed_emb
            }



