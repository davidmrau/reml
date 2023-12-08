
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate

class RAG:
    def __init__(self, retriever_kwargs, reranker_kwargs, generator_kwargs):
        self.retriever = Retrieve(retriever_kwargs )
        self.reranker = Rerank(reranker_kwargs )
        self.generator = Generate(generator_kwargs )
