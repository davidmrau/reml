def main():
    import datasets
    from rag import RAG


    experiment_folder = 'experiment_1'

    dataset = datasets.Dataset.from_dict({'sentence': ['this is the first sentence', 'desktop computer', 'soccer field', 'anatomy the docer', 'a table is set for breakfast']*3, 'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']})
    dataset.sentence_field = 'sentence'
    dataset.id_field = "id"

    def format_instruction(sample):
        docs_prompt = ''
        for i, doc in enumerate(sample['doc']):
            docs_prompt += f"Document {i}: {doc}\n"
        return f"""### Instructions: Please write a response given the query and support documents:
### Query: {sample['query']}
### Documents:
{docs_prompt}
### Response:"""

    datasets  = {
        "train": {
                "doc": dataset,
                "query": dataset,
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
            "batch_size": 3, 
            "batch_size_sim": 8,
            "top_k_documents": 3,
            }

    reranker_kwargs = {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            #"model_name": None,
            "batch_size": 3, 
            "top_k_documents": 3,
            }

    generator_kwargs = {
            #"model_name": "meta-llama/Llama-2-7b-chat-hf",
            "model_name": 'dummy',
            "batch_size": 1,
            "max_new_tokens": 1024,
            "format_instruction": format_instruction,
            }


    rag_kwargs = {
            "retriever_kwargs": retriever_kwargs, 
            "reranker_kwargs": reranker_kwargs,
            "generator_kwargs": generator_kwargs, 
            "experiment_folder": experiment_folder,
            "datasets": datasets,
    }

    rag = RAG(**rag_kwargs)
    out_generate = rag.zero_shot_single_retrieval()

    # print
    for id_, instr, response in zip(out_generate['q_id'], out_generate['response'], out_generate['instruction']):
        print('_'*40)
        print('Query id', id_)
        print('Instruction to Generator:')
        print(instr)
        print()
        print('Generated Response:')
        print(response)
        print()
        print()

    #dataset_reranker = dataset.map(reranker.retriever.tokenize, batched=True) if rerank.reranker != None else None
if __name__ == "__main__":
    main()
