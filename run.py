import hydra
from omegaconf import OmegaConf
import os

if 'CONFIG_NAME' in  os.environ:
    CONFIG_NAME = os.environ["CONFIG_NAME"]
else:
    CONFIG_NAME = 'rag'

@hydra.main(config_path="config", config_name=CONFIG_NAME)
def main(config):
    print(OmegaConf.to_yaml(config))
    from rag import RAG
    from utils import print_generate_out
    import json


    # make experiment_dir
    run_folder = f"{config.experiment_folder}/{config.run_name}"
    os.makedirs(run_folder, exist_ok=True)
    OmegaConf.save(config=config, f=f"{run_folder}/config.yaml")

    def format_instruction(sample):
        docs_prompt = ''
        for i, doc in enumerate(sample['doc']):
            docs_prompt += f"Document {i+1}: {doc}\n"
        return f"""Please write answer the query given the support documents:\n Query: {sample['query']}\n Support Documents:\n{docs_prompt}\n Response: """


    rag = RAG(**config)
    #rag.retrieve()
    out_generate = rag.generate_simple()
    if out_generate != None:
        print_generate_out(out_generate)
    print(out_generate['metrics'])

if __name__ == "__main__":
    main()
