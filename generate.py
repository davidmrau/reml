import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Generate
class Generate():
    def __init__(self, 
                 model_name=None, 
                 batch_size=1, 
                 max_new_tokens=1, 
                 format_instruction=None,
                 max_inp_length=1024
                 ):

        self.batch_size = batch_size
        self.model_name = model_name
        self.format_instruction = format_instruction
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
        self.model = generator_class(model_name=self.model_name, max_new_tokens=max_new_tokens, format_instruction=format_instruction)

    def collate_fn(self, examples):
        ids = [e['q_id'] for e in examples]
        print(examples)
        instr = [self.format_instruction(e) for e in examples]
        label = [e['label'] for e in examples]
        query = [e['query'] for e in examples]
        inp_dict = self.model.tokenizer(
            instr, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_inp_length
            )
        inp_dict['q_id'] = ids
        inp_dict['instruction'] = instr
        inp_dict['query'] = query
        inp_dict['label']= label

        return inp_dict
    
    def eval(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        responses, instructions, query_ids, queries, labels = list(), list(), list(), list(), list()

        for batch in tqdm(dataloader, desc='Generating'):
            id_ = batch.pop('q_id')
            instruction = batch.pop('instruction')
            query_ids += id_
            label = batch.pop('label')
            labels += label
            queries += batch.pop('query')
            instructions += instruction
            generated_response = self.model.generate(batch)
            responses += generated_response
        return query_ids, queries, instructions, responses, labels
