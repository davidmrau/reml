from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
import torch
from ..utils.collator import DataCollatorWithId

from collections import defaultdict

class Llama2():
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 format_instruction=None
                 ):
        self.model_name = model_name
        self.format_instruction = format_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype='float16',
                            bnb_4bit_use_dobule_quant=True
                        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', quantization_config=quant_config, use_flash_attention_2=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', quantization_config=quant_config)
        self.model.eval()
        self.model.config.pretraining_tp = 1
        self.max_new_tokens = max_new_tokens
        
    def generate(self, instr_tokenized):
        generated_ids =  self.model.generate(**instr_tokenized.to(self.device), max_new_tokens=self.max_new_tokens)
        generated_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_response

