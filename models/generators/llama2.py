from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Llama2():
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype='bfloat16',
                            bnb_4bit_use_dobule_quant=True
                        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', quantization_config=quant_config, use_flash_attention_2=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', quantization_config=quant_config)
        self.model = self.model.bfloat16()
        self.model.eval()
        self.model.config.pretraining_tp = 1
        self.max_new_tokens = max_new_tokens
        
    def generate(self, instr_tokenized):
        generated_ids =  self.model.generate(**instr_tokenized.to(self.device), max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cleaned_generated_response = [gen.split(self.get_response_marker(), 1)[1] for gen in generated_response]
        return cleaned_generated_response

    def get_response_marker(self):
        return "Answer:\n"

    def format_instruction(self, sample):
        if 'doc' in sample:
            docs_prompt = ''
            for i, doc in enumerate(sample['doc']):
                docs_prompt += f"Document {i+1}: {doc}\n"
            return f"""Please answer the question briefly given the support documents:\nQuestion: {sample['query']}\nSupport Documents:\n{docs_prompt}\n{self.get_response_marker()}"""
        else:
            return f"""Please answer the question briefly.\nQuestion: {sample['query']}\n{self.get_response_marker()}"""