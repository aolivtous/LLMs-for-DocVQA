import ast, math, random
from PIL import Image

import os, yaml, json
import argparse

import numpy as np
import torch
from SP_DocVQA import SPDocVQA



def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f)


def save_yaml(path, data):
    with open(path, 'w+') as f:
        yaml.dump(data, f)


"""
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True """

def check_config(config):
    model_name = config['model'].lower() # ADD

    if 'save_dir' in config:
        if not config['save_dir'].endswith('/'):
            config['save_dir'] = config['save_dir'] + '/'

        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])

    return True


def load_config(args):
   
    dataset_config = parse_config(yaml.safe_load(open(args.dataset, "r")), args)
    
    # Append and overwrite config values from arguments.
    # config = {'dataset_params': dataset_config, 'model_params': model_config, 'training_params': training_config}
    config = {**dataset_config}

    config = config | {k: v for k, v in args._get_kwargs() if v is not None}
    config.pop('dataset')

    # Set default seed
    """if 'seed' not in config:
        print("Seed not specified. Setting default seed to '{:d}'".format(42))
        config['seed'] = 42"""

    check_config(config)

    return config


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get('includes', []):
        config = load_config(included_config_path, args) | config

    return config


def build_dataset(config,split):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    dataset_kwargs['get_raw_ocr_data'] = True

    dataset_kwargs['use_images'] = True

    dataset_kwargs['max_pages'] = config.get('max_pages', 1)
    dataset_kwargs['hierarchical_method'] = True

    # Build dataset
    if config['dataset_name'] == 'SP-DocVQA':
        
        dataset = SPDocVQA(config['imdb_dir'], config['images_dir'], split, dataset_kwargs)

    else:
        raise ValueError

    return dataset

def extract_answers (data):
    # Iterate through the data and extract GPT answers
    gpt_answers = {}
    for item in data:
        conversation_id = item["id"].split("_")[1]
        conversations = item["conversations"]
        gpt_responses = [conv["value"] for conv in conversations if conv["from"] == "gpt"]
        gpt_answers[conversation_id] = gpt_responses
    return gpt_answers
