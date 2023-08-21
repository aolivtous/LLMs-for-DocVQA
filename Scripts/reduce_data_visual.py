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



def get_dataset_visualDocVQA(data):
    new_data = []
    length = {}

    for item in tqdm.tqdm(data):

        length[item['id']] = int(item['end_idx'])-int(item['start_idx'])
        if length[item['id']] <= 50:
            new_data.append(item)
        

  
    print(len(data))
    print(len(new_data))
    return new_data


if __name__ == "__main__":


    path = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_visualDocVQA_allContext_1.json"
    
    with open(path) as f:
        data = json.load(f)

    new_data = get_dataset_visualDocVQA(data)

    #save to json
    with open('/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_visualDocVQA_allContext_1_short50.json', 'w') as outfile:
        json.dump(new_data, outfile)