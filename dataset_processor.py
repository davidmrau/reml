import datasets
import os
from collections import defaultdict
import csv

import pickle

# doc processors

# ---------------------------------------- #
# query processors
# ---------------------------------------- #

class NQOpenProcessor:

    @staticmethod
    def process(split, num_proc=None):

        hf_name = 'nq_open' 
        dataset = datasets.load_dataset(hf_name, num_proc=num_proc)[split]


        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        dataset = dataset.rename_column("answer", "label")
        dataset = dataset.rename_column("question", "content")

        return dataset
    
# ---------------------------------------- #
# collection processors
# ---------------------------------------- #

class ODQAWikiCorpora100WTamberProcessor:

    @staticmethod
    def process(split, num_proc=None):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_subset_name= "wiki-text-100w-tamber"

        dataset = datasets.load_dataset(hf_name, hf_subset_name, num_proc=num_proc)[split]
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=num_proc)
        dataset = dataset.rename_column("docid", "id")
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset
    

class ODQAWikiCorpora100WKarpukhinProcessor:

    @staticmethod
    def process(split, num_proc=None):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_subset_name= "wiki-text-100w-karpukhin"

        dataset = datasets.load_dataset(hf_name, hf_subset_name, num_proc=num_proc)[split]
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=num_proc)
        dataset = dataset.rename_column("docid", "id")
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset
    
class ODQAWikiCorpora63tamberProcessor:

    @staticmethod
    def process(split, num_proc=None):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_subset_name= "wiki-text-6-3-tamber"

        dataset = datasets.load_dataset(hf_name, hf_subset_name, num_proc=num_proc)[split]
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=num_proc)
        dataset = dataset.rename_column("docid", "id")
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset


class Processor(object):
    
    @staticmethod
    def process():
        raise NotImplementedError()
    
    def add_index(self, dataset):
        dataset = dataset.add_column("index", range(len(dataset)))    
        return dataset
    
    def get_index_to_id(self, dataset):
        if 'index' not in dataset.features:
            dataset = self.add_index(dataset)
        return dict(zip(dataset["id"], dataset["index"]))
    
    def get_dataset(self, dataset_name, split, debug=False):
        print(f'Processing dataset: {dataset_name}')
        out_folder = f'{self.out_folder}/{dataset_name}_{split}'
        if os.path.exists(out_folder) and not self.overwrite:
            dataset = datasets.load_from_disk(out_folder)
            if debug:
                dataset = dataset.select(range(3))
            #id2index = self.tsv_to_dict(f'{out_folder}/id2index.csv')
            id2index = pickle.load(open(f'{out_folder}/id2index.p', 'rb'))
            dataset.id2index = id2index
        else:
            dataset = self.processors[dataset_name].process(split, num_proc=self.num_proc)
            dataset.save_to_disk(out_folder)
            id2index = self.get_index_to_id(dataset) 
            dataset.id2index = id2index
            pickle.dump(id2index, open(f'{out_folder}/id2index.p', 'wb'))
            #self.dict_to_tsv(id2index, f'{out_folder}/id2index.csv')
        dataset.name = dataset_name
        
        return dataset
    
    def dict_to_tsv(self, id_to_index, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                # Write data
                for id, index in id_to_index.items():
                    row = f"{id}\t{index}\n"
                    file.write(row)
        except Exception as e:
            print(f"Error writing id2index file: {e}")

    def tsv_to_dict(self, file_path):
        try:
            id_to_index = {}
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    if len(row) == 2:
                        id, index = row
                        id_to_index[id] = int(index)
            return id_to_index
        except Exception as e:
            print(f"Error loading id2index file: {e}")
            return None



class CollectionProcessor(Processor):
    def __init__(self, out_folder='datasets', num_proc=20, overwrite=False):
        self.num_proc = num_proc
        self.out_folder = out_folder
        self.overwrite = overwrite
        self.processors = {
            "odqa-wiki-corpora-100w-tamber": ODQAWikiCorpora100WTamberProcessor,
            "odqa-wiki-corpora-100w-karpukhin": ODQAWikiCorpora100WKarpukhinProcessor,
        }


class QueryProcessor(Processor):
    def __init__(self, out_folder='datasets', num_proc=20, overwrite=False):
        self.num_proc = num_proc
        self.out_folder = out_folder
        self.overwrite = overwrite
        self.processors = defaultdict(None)
        self.processors.update({
            "nq_open": NQOpenProcessor,
        })

# applies processing to dataset names
# processes query and doc with different processors

class ProcessDatasets:
    @staticmethod
    def process(datasets, out_folder='datasets', num_proc=1, overwrite=False, debug=False):
        processed_datasets = defaultdict(dict)
        doc_processor = CollectionProcessor(out_folder=out_folder, num_proc=num_proc, overwrite=overwrite)
        query_processor = QueryProcessor(out_folder=out_folder, num_proc=num_proc, overwrite=overwrite)
        for split in datasets:
            for subset in datasets[split]:
                if datasets[split][subset] != None:
                    dataset_name, dataset_split_name = datasets[split][subset]
                    if subset == 'query':
                        processor_class = query_processor
                        dataset = processor_class.get_dataset(dataset_name, dataset_split_name, debug=debug)
                    elif subset == 'doc':
                        processor_class = doc_processor
                        dataset = processor_class.get_dataset(dataset_name, dataset_split_name, debug=False)
                    processed_datasets[split][subset] = dataset
                else:
                    processed_datasets[split][subset] = None
        return processed_datasets

    
