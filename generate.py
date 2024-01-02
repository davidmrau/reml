import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Generate
class Generate():
    def __init__(self, 
                 model_name=None, 
                 batch_size=1, 
                 max_new_tokens=1, 
                 max_inp_length=1024
                 ):

        self.batch_size = batch_size
        self.model_name = model_name
        self.max_inp_length = max_inp_length

        #if self.batch_size > 1:
        #    raise NotImplementedError('Only batch size 1 is implemented yet.')
        if self.model_name == 'dummy':
            from models.generators.dummy import Dummy
            generator_class = Dummy
        elif self.model_name == 'meta-llama/Llama-2-7b-chat-hf':
            from models.generators.llama2 import Llama2
            generator_class = Llama2
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented!")

        # instatiate model
        self.model = generator_class(model_name=self.model_name, max_new_tokens=max_new_tokens)

    def collate_fn(self, examples):
        ids = [e['q_id'] for e in examples]
        instr = [self.model.format_instruction(e) for e in examples]
        label = [e['label'] for e in examples]
        query = [e['query'] for e in examples]
        inp_dict = self.model.tokenizer(
            instr, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_inp_length
            )
        info_dict = {}
        info_dict['q_id'] = ids
        info_dict['instruction'] = instr
        info_dict['query'] = query
        info_dict['label']= label
        return inp_dict, info_dict
    
    def eval(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        responses, instructions, query_ids, queries, labels = list(), list(), list(), list(), list()

        for inp_dict, info_dict in tqdm(dataloader, desc='Generating'):
            id_ = info_dict['q_id']
            instruction = info_dict['instruction']
            query_ids += id_
            label = info_dict['label']
            labels += label
            queries += info_dict['query']
            instructions += instruction

            if hasattr(self.model, 'gold_answer'):
                generated_response = self.model.gold_answer(label)
            else:
                generated_response = self.model.generate(inp_dict)
            
            responses += generated_response
        return query_ids, queries, instructions, responses, labels
