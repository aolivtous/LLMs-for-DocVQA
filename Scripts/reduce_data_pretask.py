import argparse
import json
import os
import random
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from utils import build_dataset, load_config
from transformers import CLIPImageProcessor 




def get_data():
    #read json
    path = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced.json"
    path_2 = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData.json"

    with open(path) as f:
        data = json.load(f)
    with open(path_2) as f:
        data_2 = json.load(f)
    ids = []
    for d in data:
        ids.append(d['id'])

    length = len(data)
    test = []
    val = []
    #randomly put half of the data in test list and the other half in val list
    #shuffle data
    random.shuffle(data)

    for t in data_2:
        if t['id'] not in ids:
            test.append(t)
    
    #shuffle test   
    random.shuffle(test)
    #get 1000
    test = test[:1000]

    with open('/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced_test_1000.json', 'w') as outfile:
        json.dump(test, outfile)
    

if __name__ == "__main__":


    """path = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_visualDocVQA_allContext_1.json"
    
    with open(path) as f:
        data = json.load(f)

    new_data = get_dataset_visualDocVQA(data)

    #save to json
    with open('/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_visualDocVQA_allContext_1_short50.json', 'w') as outfile:
        json.dump(new_data, outfile)"""

    get_data()