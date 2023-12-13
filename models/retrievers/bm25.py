from pyserini.search import LuceneSearcher
import subprocess
from tqdm import tqdm
import os
import json
from torch.utils.data import DataLoader


class BM25:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, dataset, index_path=None,  top_k_documents=1, prebuild_index=None, batch_size=256, num_threads=1, return_docs=False):
        self.num_threads = num_threads
        if prebuild_index:
            print(f'Loading Pre-build Index: {prebuild_index}')
            model = LuceneSearcher.from_prebuilt_index(prebuild_index)
        else:
            model = LuceneSearcher(index_path)
        # Batch search
        q_ids, doc_ids, scores, docs = list(), list(), list(), list()
        for examples in tqdm(dataset.iter(batch_size=batch_size), desc=f'Retrieving with {num_threads} threads.'):
            q_id_batch, query_batch = examples['id'], examples['content']
            all_hits = model.batch_search(queries=query_batch, qids=q_id_batch, k=top_k_documents, threads=self.num_threads)
            for q_id, hits in all_hits.items():
                q_scores, q_doc_ids, q_docs = list(), list(), list()
                for hit in hits:
                    doc_id = hit.docid
                    score = hit.score
                    if return_docs:
                        doc = json.loads(model.doc(doc_id).get("raw"))['contents']
                        q_docs.append(doc)
                    q_doc_ids.append(doc_id)
                    q_scores.append(score)
                doc_ids.append(q_doc_ids)
                scores.append(q_scores)
                q_ids.append(q_id)
                if return_docs:
                    docs.append(q_docs)
                

        return {
            "q_id": q_ids,
            "doc_id": doc_ids, 
            "score": scores,
            "doc": docs if return_docs else None
        }

    def index(self, dataset, dataset_path):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            json_dir = f'{dataset_path}_json/'
            self.save_documents_to_json(dataset, json_dir) 
            self.run_index_process(dataset_path, json_dir)
        return 

    def run_index_process(self, out_dir, json_dir):
        command = [
            'python3', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', json_dir,
            '--index', out_dir,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', self.num_threads
            # '--storePositions', '--storeDocvectors', '--storeRaw'
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    def save_documents_to_json(self, dataset, output_dir, max_per_file=100000):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Split the dataset into chunks
        contents = [dataset['content'][i:i + max_per_file] for i in range(0, len(dataset), max_per_file)]
        ids = [dataset['id'][i:i + max_per_file] for i in range(0, len(dataset), max_per_file)]
        # Save each chunk to a separate JSON file
        for i, (chunk_id, chunk_sent) in tqdm(enumerate(zip(ids, contents), start=1), desc='Saving dataset to json.'):
            formatted_chunk = [{"id": id_, "contents": sent} for id_ , sent in zip(chunk_id, chunk_sent)]
            output_file_path = os.path.join(output_dir, f"doc{i}.json")
            with open(output_file_path, "w") as output_file:
                json.dump(formatted_chunk, output_file, indent=2)
            print(f"Saved {len(formatted_chunk)} documents to {output_file_path}")




# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input tests/resources/sample_collection_jsonl \
#   --index indexes/sample_collection_jsonl \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 1 \
#   --storePositions --storeDocvectors --storeRaw
