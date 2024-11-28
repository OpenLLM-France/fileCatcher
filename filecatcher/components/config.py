import os
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra import initialize, compose

def load_config(config_path="../.hydra_config", overrides=None)-> OmegaConf:     
    with initialize(config_path=config_path, job_name="config_loader"):
        config = compose(config_name="config", overrides=overrides)
        config.paths.root_dir = Path(config.paths.root_dir).absolute()
        return config