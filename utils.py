import datasets
import random 
import json
from collections import defaultdict

def get_by_ids(dataset, ids):
    # if single id is passed cast it to list
    if not isinstance(ids, list):
        ids = [ids]
    idxs = [ dataset.id2index[id_] for id_ in ids]
    return dataset[idxs]['content']


# gets q_ids and d_ids and does a lookup by id to get the content
# then constructs hf_dataset out of it
# pyserini_docs=True documents have been returned by bm25 and don't contents don't need to be fetched
def make_hf_dataset(dataset, q_ids, d_ids, multi_doc=False, pyserini_docs=None):

    if q_ids == d_ids == None:
        dataset_dict = {
            'query': dataset['query']['content'], 
            'q_id': dataset['query']['id'],
            'label': dataset['query']['label'],
            }
    else:
        dataset_dict = {'query': [], 'doc': [], 'q_id': []}
        labels = dataset['query']['label'] if 'label' in dataset['query'].features else None
        if labels != None:
            dataset_dict['label'] = []
        if not multi_doc:
            dataset_dict['d_id'] = []
        assert len(d_ids) == len(q_ids)

        queries = get_by_ids(dataset['query'], q_ids)
        for i, q_id in enumerate(q_ids):
            if pyserini_docs == None:
                docs = get_by_ids(dataset['doc'], d_ids[i])
            else:
                docs = pyserini_docs[i]
            # for multi_doc=True, all documents are saved to the 'doc' entry
            if multi_doc:
                dataset_dict['doc'].append(docs)
                dataset_dict['query'].append(queries[i])
                dataset_dict['q_id'].append(q_id)
                if labels != None:
                    dataset_dict['label'].append(labels[i])
            else:
                # for multi_doc=False, we save every document to a new entry
                    for d_id, doc in zip(d_ids[i], docs):
                        dataset_dict['d_id'].append(d_id)
                        dataset_dict['doc'].append(doc)
                        dataset_dict['query'].append(queries[i])
                        dataset_dict['q_id'].append(q_id)
                        if labels != None:
                            dataset_dict['label'].append(labels[i])
    return datasets.Dataset.from_dict(dataset_dict)

def print_generate_out(queries, instructions, responses, query_ids, labels, n=3):
    rand = random.sample(range(len(query_ids)), n)
    for i in rand:
        print('_'*50)
        print('Query ID:', query_ids[i])
        print('Query:', queries[i])
        print('_'*50)
        print('Instruction to Generator:')
        print(instructions[i])
        print()
        print('Generated Response:')
        print(responses[i])
        print('Label:')
        print(labels[i])
        print()
        print()


def print_rag_model(rag, retriever_kwargs,reranker_kwargs, generator_kwargs):
    print(':'*100)
    print('RAG Model:')
    # init modules
    if retriever_kwargs != None:
        print(f"Retriever: {retriever_kwargs['model_name']}")
    if reranker_kwargs != None:
        print(f"Reranker: {reranker_kwargs['model_name']}")
    if generator_kwargs != None:
        print(f"Generator: {generator_kwargs['model_name']}")

    print(':'*100)
    print()


def write_trec(fname, q_ids, d_ids, scores):
    with open(fname, 'w') as fout:
        for i, q_id in enumerate(q_ids):
            for rank, (d_id, score) in enumerate(zip(d_ids[i], scores[i])):
                fout.write(f'{q_id}\tq0\t{d_id}\t{rank+1}\t{score.item()}\trun\n')

def write_generated(out_folder, query_ids, instructions, responses, labels, metrics_out, generation_time_seconds):
    json_dict = {}
    json_dict['gen_time_hours'] = generation_time_seconds/3600
    json_dict['metrics'] = metrics_out
    json_dict['generated'] = list()
    
    for i, (q_id, response, instruction, label) in enumerate(zip(query_ids, responses, instructions, labels)):
        jsonl = {}
        jsonl['q_id'] = q_id
        jsonl['response'] = response
        jsonl['instruction'] = instruction
        jsonl['label'] = label
        json_dict['generated'].append(jsonl)

    with open(f'{out_folder}/generated_out.json', 'w') as fp:
        json.dump(json_dict, fp)

def load_trec(fname):
    # read file
    trec_dict = defaultdict(list)
    for l in open(fname):
        q_id, _, d_id, _, score, _ = l.split('\t')
        trec_dict[q_id].append((d_id, score))
    q_ids, d_ids, scores = list(), list(), list()
    for q_id in trec_dict:
        q_ids.append(q_id)
        d_ids_q, scores_q = list(), list()
        for d_id, score in trec_dict[q_id]:
            d_ids_q.append(d_id)
            scores_q.append(float(score))
        d_ids.append(d_ids_q)
        scores.append(scores_q)
    return q_ids, d_ids, scores