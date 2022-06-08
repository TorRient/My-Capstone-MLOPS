import numpy as np
import yaml
import argparse
from PIL import Image
import tqdm
import sys
sys.path.append("./")
from src.tool.utils import compute_accuracy
from src.tool.predictor import Predictor
from src.tool.config import Cfg

from dvclive import Live

dvclive = Live()

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def loader(path_nano):
    file_nano = open(path_nano, "r")
    l_img = []
    l_text = []
    for i, line in enumerate(file_nano):
        path_img, text = line.replace("\n", "").split("\t")
        l_img.append(path_img)
        l_text.append(text)
        if i == 500:
            break
    return l_img, l_text

def eval(path_nano, config_yaml):
    l_img, l_text = loader(path_nano)
    print(f"Found {len(l_img)} images")
    batch = 16
    iter = len(l_img)//batch
    inputs = []
    predicts = []
    i = 0
    if iter > 0:
        for i in tqdm.tqdm(range(iter)):
            for j in range(i*batch,(i+1)*batch):
                img = Image.open(root_dir + l_img[j])
                inputs.append(img)
            results = detector.predict_batch(inputs)
            inputs = []
            predicts += results
    inputs = []
    for j in range((i+1)*16, len(l_img)):
        img = Image.open(root_dir + l_img[j])
        inputs.append(img)
    if len(inputs) != 0:
        results = detector.predict_batch(inputs)
        predicts += results

    acc_full_seq = compute_accuracy(l_text, predicts, mode='full_sequence')
    acc_per_char = compute_accuracy(l_text, predicts, mode='per_char')
    dvclive.log("Acc full sequence", acc_full_seq)
    dvclive.log("Acc per character", acc_per_char)
    return acc_full_seq, acc_per_char

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config_yaml = read_params(parsed_args.config)
    root_dir = "./"
    config = Cfg.load_config_from_file(config_yaml["model_name"])
    detector = Predictor(config)
    anno_dataset = config_yaml["anno_dataset"]
    eval(root_dir + f"{anno_dataset}.txt", config)