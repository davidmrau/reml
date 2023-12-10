# reranking
class Rerank():
    def __init__(self, model_name=None, datasets=None):
        self.model_name = model_name
        if self.model_name == None:
            self.model = None
