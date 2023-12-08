# reranking
class Rerank():
    def __init__(self, kwargs):

        if kwargs['model_name'] == None:
            print('Not using Reranker!')
            self.reranker = None
