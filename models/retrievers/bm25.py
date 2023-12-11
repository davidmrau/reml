from pyserini.search import LuceneSearcher
import subprocess
from tqdm import tqdm
import os
import json
from torch.utils.data import DataLoader
class BM25:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, dataset, index_path=None,  top_k_documents=1):
        model = LuceneSearcher(index_path)
        # Batch search
        q_ids, doc_ids, scores = list(), list(), list()
        for example in dataset:
            q_id, query = example['id'], example['sentence']
            hits = model.search(query, k=top_k_documents)
            for hit in hits:
                doc_id = hit.docid
                score = hit.score
            doc_ids.append(doc_id)
            scores.append(score)
            q_ids.append(q_id)

        return q_ids, doc_ids, scores

    def index(self, dataset, dataset_path):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            json_dir = f'{dataset_path}_json/'
            self.save_documents_to_json(dataset, json_dir) 
            self.run_index_process(dataset_path, json_dir)
        return 

    def run_index_process(self, out_dir, json_dir, threads='8'):
        command = [
            'python3', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', json_dir,
            '--index', out_dir,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', threads
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
        sentences = [dataset['sentence'][i:i + max_per_file] for i in range(0, len(dataset), max_per_file)]
        ids = [dataset['id'][i:i + max_per_file] for i in range(0, len(dataset), max_per_file)]
        # Save each chunk to a separate JSON file
        for i, (chunk_id, chunk_sent) in tqdm(enumerate(zip(ids, sentences), start=1), desc='Saving dataset to json.'):
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
