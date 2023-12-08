from transformers import AutoModel
class Splade:

    def __init__(self, model_name):

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, device_map='auto', low_cpu_mem_usage=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def search(self):
        return
