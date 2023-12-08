from models.generators.dummy import Dummy
from models.generators.llama2 import Llama2

# Generate
class Generate():
    def __init__(self, kwargs):
        self.model_name = kwargs['model_name']

        # match model class
        if self.model_name == None:
            print('Not using a Generator!')
            self = None
        elif self.model_name == 'dummy':
            generator_class = Dummy
        elif self.model_name == 'meta-llama/Llama-2-7b-chat-hf':
            generator_class =
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented!")

        # instatiate model
        self.generator = generator_class(kwargs)


    def generate(self, dataset_q, dataset_doc, query_ids, doc_ids, q_map, d_map):

        assert len(doc_ids) == len(query_ids)
        q_idxs = [ q_map[id_] for id_ in query_ids]
        queries = dataset_q[q_idxs]['sentence']
        responses = list()
        instructions = list()
        for i in range(len(query_ids)):
            d_idxs = [ d_map[id_] for id_ in doc_ids[i]]
            docs = dataset_doc[d_idxs]
            instruction, response  = self.generator.generate(queries[i], docs)
            responses.append(response)
        return {
                "instructions": instructions,
                "responses": responses
            }
