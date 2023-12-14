def main():
    from rag import RAG
    from utils import print_generate_out, get_by_ids
    import json


    def format_instruction(sample):
        docs_prompt = ''
        for i, doc in enumerate(sample['doc']):
            docs_prompt += f"Document {i}: {doc}\n"
        return f"""### Instructions: Please write a response given the query and support documents:\n### Query: {sample['query']}\n### Documents:{docs_prompt}\n### Response:"""


    dataset_names  = {
        "train": {
                "doc": None,
                "query": None,
            },
        "test": {
                "doc": ("odqa-wiki-corpora-100w-karpukhin", "train"),
                "query": ("nq_open", "validation"),
            },
        "dev": {
                "doc": None,
                "query": None,
            },
    }

    prebuild_indexes = json.loads(open('config/prebuild_indexes.json').read())

    retriever_kwargs = {
            #"model_name": "facebook/contriever",
            "model_name": "bm25",
            "batch_size": 1024,
            "batch_size_sim": 256,
            "top_k_documents": 10,
            "pyserini_num_threads": 10, 
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
            "max_inp_length": None
            }

    rag_kwargs = {
            "retriever_kwargs": retriever_kwargs,
            "reranker_kwargs": reranker_kwargs,
            "generator_kwargs": generator_kwargs,
            "run_name": 'test',
            "dataset_names": dataset_names,
            "dataset_folder": '/scratch-shared/drau_datasets/',
            "index_folder": '/scratch-shared/drau_indexes/',
            "experiment_folder": '/scratch-shared/drau_experiments',
            "processing_num_proc": 30, 
            "overwrite_datasets": False,
            "prebuild_indexes": prebuild_indexes,
    }

    rag = RAG(**rag_kwargs)
    #rag.retrieve()
    out_generate = rag.generate_simple()
    if out_generate != None:
        print_generate_out(out_generate)
    print(out_generate['metrics'])

if __name__ == "__main__":
    main()
