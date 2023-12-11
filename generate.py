import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Generate
class Generate():
    def __init__(self, model_name=None, batch_size=1, max_new_tokens=1, format_instruction=None):

        self.batch_size = batch_size
        self.model_name = model_name
        self.format_instruction = format_instruction

        if self.batch_size > 1:
            raise NotImplementedError('Only batch size 1 is implemented yet.')
        # match model class
        if self.model_name == None:
            self = None
        else:
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

    def eval(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.model.collate_fn)
        responses = list()
        instructions = list()
        query_ids = list()
        for batch in tqdm(dataloader, desc='Generating'):
            id_ = batch.pop('q_id')
            instruction = batch.pop('instruction')
            query_ids += id_
            instructions += instruction
            generated_response = self.model.generate(instruction)
            responses += generated_response
        return query_ids, instructions, responses
