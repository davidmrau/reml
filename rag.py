
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate

class RAG:
    def __init__(self, retrieve=None, rerank=None, generate=None):

        self.retrieve = retrieve
        self.rerank = rerank
        self.generate = generate
        self.modules = {'gen': self.generate, 'ret': self.retrieve, 'rerank': self.rerank}

    def process_data(self):
        for mod in self.modules:
            print(f'Processing dataset for {mod}...')
            mod.eval_dataset_q = mod.eval_dataset_q.map(mod.retriever.tokenize, batched=True, fn_kwargs={"sentence_field": mod.eval_dataset_q.sentence_field, "id_field": mod.eval_dataset_q.id_field})
            mod.eval_dataset_doc = mod.eval_dataset_doc.map(mod.retriever.tokenize, batched=True, fn_kwargs={"sentence_field": mod.eval_dataset_doc.sentence_field, "id_field": mod.eval_dataset_doc.id_field})
            mod.eval_dataset_q = mod.eval_dataset_q.remove_columns(['sentence'])
            mod.eval_dataset_doc = mod.eval_dataset_doc.remove_columns(['sentence'])
        # init datasets retriever
        
        


    def run(self):
        out_retrieve = self.retrieve.eval(return_embeddings=False)
        print('out_retrieve', out_retrieve)
        # "doc_embs"
        # "q_embs"
        # "scores"
        # "q_ids"
        # "doc_ids
        out_generate = self.generate.eval(out_retrieve['q_ids'], out_retrieve['doc_ids'])
        for instr, reponse in zip(out_generate['responses'], out_generate['instructions']):
            print('Instruction to LLM:')
            print(instr)
            print()
            print('Generated Response:')
            print(response)
            print()
            print()
