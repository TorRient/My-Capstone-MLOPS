import argparse
import sys
import yaml
sys.path.append("./")
from src.models.trainer import Trainer
from src.tool.config import Cfg

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def main(config_yaml):
    config = Cfg.load_config_from_file(config_yaml["model_name"])

    trainer = Trainer(config)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config_yaml = read_params(parsed_args.config)
    main(config_yaml)