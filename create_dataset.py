import yaml
import argparse
from glob import glob

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def create_data(data_path):
    list_test = glob(data_path + "/*")
    with open(data_path + ".txt", "w") as files:
        for path_img in list_test:
            if path_img.split(".")[-1] != "txt":
                with open(path_img.split(".")[0] + ".txt", "r") as file_anno:
                    text = file_anno.read()
                files.write(path_img + "\t" + text + "\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(parsed_args.config)
    create_data(config["anno_dataset"])
