def main():
    import datasets
    #from rag import RAG

    from retrieve import Retrieve
    from rerank import Rerank
    from generate import Generate

    from utils import LookupDatasetHF

    retriever_kwargs = {
            "model_name": "facebook/contriever",
            "batch_size": 3, "batch_size_sim": 8,
            'top_k_documents': 3
            }
    reranker_kwargs = {
            "model_name": None
            }
    generator_kwargs = {
            "model_name": "dummy",
            "batch_size": 1
            }

    retrieve = Retrieve(retriever_kwargs)
    rerank = Rerank(reranker_kwargs)
    generate = Generate(generator_kwargs)

    dataset = datasets.Dataset.from_dict({'sentence': ['this is the first sentence', 'desktop computer', 'soccer field', 'anatomy the docer', 'a table is set for breakfast']*3, 'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']})
    dataset.sentence_field = 'sentence'
    dataset.id_field = "id"
    # init datasets retriever
    dataset_retriever = dataset.map(retrieve.retriever.tokenize, batched=True, fn_kwargs={"sentence_field": dataset.sentence_field, "id_field": dataset.id_field}) if retrieve.retriever.model_name != 'BM25' else dataset
    dataset_retriever = dataset_retriever.remove_columns(['sentence'])

    queries_dataset_retriever = dataset_retriever.select(range(3))
    docs_dataset_retriever = dataset_retriever


    out_retrieve = retrieve.retrieve(queries_dataset_retriever, docs_dataset_retriever, return_embeddings=False)
    print('out_retrieve', out_retrieve)
    # "doc_embs"
    # "q_embs"
    # "scores"
    # "q_ids"
    # "doc_ids

    dataset = datasets.Dataset.from_dict({'sentence': ['this is the first sentence', 'desktop computer', 'soccer field', 'anatomy the docer', 'a table is set for breakfast']*3, 'id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']})
    dataset.sentence_field = 'sentence'
    dataset.id_field = "id"

    queries_dataset_generator = dataset.select(range(3))
    docs_dataset_generator = dataset


    queries_dataset_generator = queries_dataset_generator.add_column("index", range(len(queries_dataset_generator)))
    q_map = dict(zip(queries_dataset_generator["id"], queries_dataset_generator["index"]))

    docs_dataset_generator = docs_dataset_generator.add_column("index", range(len(docs_dataset_generator)))
    d_map = dict(zip(docs_dataset_generator["id"], docs_dataset_generator["index"]))


    out_generate = generate.generate(queries_dataset_generator, docs_dataset_generator, out_retrieve['q_ids'], out_retrieve['doc_ids'], q_map, d_map)
    for instr, reponse in zip(out_generate['responses'], out_generate['instructions']):
        print('Instruction to LLM:')
        print(instr)
        print()
        print('Generated Response:')
        print(response)
        print()
        print()
    #dataset_reranker = dataset.map(reranker.retriever.tokenize, batched=True) if rerank.reranker != None else None
if __name__ == "__main__":
    main()
