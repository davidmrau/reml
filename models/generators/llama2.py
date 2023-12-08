from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Llama2():
    def __init__(self, kwargs):
        self.model_name = kwargs['model_name']
        #self.top_k_documents = kwargs['top_k_documents']
        if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
            raise NotImplemtedError('Only batch size 1 is implemented yet.')
        self.batch_size = kwargs['batch_size']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype='float16',
                            bnb_4bit_use_dobule_quant=True
                        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', quantization_config=quant_config,use_flash_attention_2=True)

        self.model.config.pretraining_tp = 1
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.model.embed_tokens.padding_idx = len(tokenizer) - 1
        self.model.model.embed_tokens._fill_padding_idx_with_zero()

        self.model.config.use_cache = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, query, documents):
        instrucion = self.format_instruction(query, documents)
        instr_tokenized = tokenizer(instrucion, padding=True, truncation=True, return_tensors="pt", max_length=1024)
        with torch.no_grad():
            output = self.model.generate(**instr_tokenized.to(self.device), max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
        generated_ids = model.generate(inputs.input_ids, max_length=30)

        generated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(generated_response)
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
