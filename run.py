def main():
    import datasets
    from rag import RAG
    from utils import print_generate_out

    dataset = datasets.Dataset.load_from_disk('datasets/dummy')

    def format_instruction(sample):
        docs_prompt = ''
        for i, doc in enumerate(sample['doc']):
            docs_prompt += f"Document {i}: {doc}\n"
        return f"""### Instructions: Please write a response given the query and support documents:\n### Query: {sample['query']}\n### Documents:{docs_prompt}\n### Response:"""

    datasets  = {
        "train": {
                "doc": None,
                "query": None,
            },
        "eval": {
                "doc": dataset,
                "query": dataset,
            },
        "test": {
                "doc": None,
                "query": None,
            },
    }

    retriever_kwargs = {
            "model_name": "facebook/contriever",
            #"model_name": "bm25",
            "batch_size": 3,
            "batch_size_sim": 8,
            "top_k_documents": 3,
            }

    reranker_kwargs = {
            #"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_name": None,
            "batch_size": 3,
            "top_k_documents": 2,
            }

    generator_kwargs = {
            #"model_name": "meta-llama/Llama-2-7b-chat-hf",
            "model_name": 'dummy',
           # "model_name": None,
            "batch_size": 1,
            "max_new_tokens": 1024,
            "format_instruction": format_instruction,
            }

    rag_kwargs = {
            "retriever_kwargs": retriever_kwargs,
            "reranker_kwargs": reranker_kwargs,
            "generator_kwargs": generator_kwargs,
            "experiment_folder": 'experiments',
            "run_name": 'test',
            "datasets": datasets,
    }

    rag = RAG(**rag_kwargs)
    #rag.retrieve()
    out_generate = rag.generate_simple()
    if out_generate != None:
        print_generate_out(out_generate)

if __name__ == "__main__":
    main()
