def main():
    import datasets
    from rag import RAG

    from retrieve import Retrieve
    from rerank import Rerank
    from generate import Generate

    from utils import LookupDatasetHF
    print('startup')


    

    dataset = datasets.Dataset.from_dict({'sentence': ['this is the first sentence', 'desktop computer', 'soccer field', 'anatomy the docer', 'a table is set for breakfast']*3, 'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']})
    dataset.sentence_field = 'sentence'
    dataset.id_field = "id"

    retriever_kwargs = {
            "model_name": "facebook/contriever",
            "batch_size": 3, "batch_size_sim": 8,
            "top_k_documents": 3,
            "eval_dataset_q": dataset, 
            "eval_dataset_doc": dataset, 
            }
    
    reranker_kwargs = {
            "model_name": None
            }
    
    generator_kwargs = {
            #"model_name": "meta-llama/Llama-2-7b-chat-hf",
            "model_name": 'dummy',
            "batch_size": 1,
            "eval_dataset_q": dataset, 
            "eval_dataset_doc": dataset, 
            }
    

    print('Loading retriever...')
    retrieve = Retrieve(retriever_kwargs)

    rerank = Rerank(reranker_kwargs)
    print('Loading generator...')
    generate = Generate(generator_kwargs)

    rag = RAG(retrieve=retrieve, rerank=rerank, generate=generate)
    rag.run()

    #dataset_reranker = dataset.map(reranker.retriever.tokenize, batched=True) if rerank.reranker != None else None
if __name__ == "__main__":
    main()
