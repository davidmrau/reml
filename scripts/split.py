import glob 
import torch
import os
from tqdm import tqdm

def split_sparse_tensors(input_folder, output_folder, num_chunks):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    files = glob.glob(f'{input_folder}/*.pt')
    input_files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))


    # List all files in the input folder and sort them

    # A list to store chunks for each input file
    all_chunks = []

    for j, input_file in tqdm(enumerate(input_files)):
        torch.cuda.empty_cache()
        # Load the sparse tensor
        #input_path = os.path.join(input_folder, input_file)
        sparse_tensor = torch.load(input_file).to_dense()

        # Calculate the target chunk size and remainder
        chunk_size = sparse_tensor.shape[0] // num_chunks
        remainder = sparse_tensor.shape[0] % num_chunks

        # Split the sparse tensor into smaller chunks
        chunks = []
        start_idx = 0
        for i in range(num_chunks):
            # Adjust the chunk size for the remainder
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            # Properly slice the sparse tensor
            chunk = sparse_tensor[start_idx:end_idx, :].to_sparse()
            start_idx = end_idx
            output_file = f"embedding_chunk_{j + 1}_{i + 1}.pt"
            output_path = os.path.join(output_folder, output_file)
            torch.save(chunk, output_path)

        del sparse_tensor
        del chunk

if __name__ == "__main__":
    input_folder = "indexes/odqa-wiki-corpora-100w-karpukhin_doc_naver_splade-cocondenser-selfdistil/"
    output_folder = "indexes/odqa-wiki-corpora-100w-karpukhin_doc_naver_splade-cocondenser-selfdistil_smaller_chunks/"
    num_chunks = 5

    split_sparse_tensors(input_folder, output_folder, num_chunks)


