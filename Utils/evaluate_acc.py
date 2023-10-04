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
    path = "/home/aolivera/TFM-LLM/LLM/Results/evaluation/vicuna7B_val_latin_spaces_2023-06-20_17-54-56.json"

    with open(path) as f:
        data = json.load(f)

    acc = 0
    
    keys = data["Scores by samples"].keys()
    for key in keys:
        
       
        if data["Scores by samples"][key]["anls"] == 1:
            acc += 1
    
    acc = acc/len(keys)
    print("The accuracy is = ", acc)

    
    

if __name__ == "__main__":


    get_data()