import hydra
from multiprocess import set_start_method
import os

if 'CONFIG_NAME' in  os.environ:
    CONFIG_NAME = os.environ["CONFIG_NAME"]
else:
    CONFIG_NAME = 'rag'

@hydra.main(config_path="config", config_name=CONFIG_NAME, version_base="1.2")
def main(config):
    from rag import RAG
    rag = RAG(**config, config=config)
    rag.default(split='test')
if __name__ == "__main__":
    # needed for multiprocessing to avoid CUDA forked processes erro
    # https://huggingface.co/docs/datasets/main/en/process#multiprocessing
    set_start_method("spawn")
    main()
