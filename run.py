import hydra
from multiprocess import set_start_method
from omegaconf import OmegaConf
import os

if 'CONFIG_NAME' in  os.environ:
    CONFIG_NAME = os.environ["CONFIG_NAME"]
else:
    CONFIG_NAME = 'rag'

@hydra.main(config_path="config", config_name=CONFIG_NAME, version_base="1.2")
def main(config):
    print(OmegaConf.to_yaml(config))
    from rag import RAG
    import json

    # make dirs
    os.makedirs(config.experiment_folder, exist_ok=True)
    os.makedirs(config.index_folder, exist_ok=True)
    os.makedirs(config.run_folder, exist_ok=True)
    run_folder = f"{config.experiment_folder}/{config.run_name}"
    os.makedirs(run_folder, exist_ok=True)
    OmegaConf.save(config=config, f=f"{run_folder}/config.yaml")

    rag = RAG(**config)
    rag.default(split='test')
if __name__ == "__main__":
    # needed for multiprocessing to avoid CUDA forked processes erro
    # https://huggingface.co/docs/datasets/main/en/process#multiprocessing
    set_start_method("spawn")
    main()
