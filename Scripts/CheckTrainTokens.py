import argparse
import json
import tqdm
import torch
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from LLM.Scripts.evaluation_DocVQA import parse_args
from utils import build_dataset, load_config


if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file', type=str, default='/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/preprocessed_data/train_processed_text_BB.json', help='path of the output JSON file that saves the context + questions in the FastChat format and the answers')
    
    args = parser.parse_args()

    #load npy file
    config = load_config(parse_args())


    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    
    i = 0
    lengths = []
    for ids in data['input_ids']:        
        lengths.append(len(ids))
    
 
    #find the lengths => 2048
    i = 0
    print("length of the tokens ", len(lengths))
    print("checking the lengths")
    for length in lengths:
        if length > 2048:
            #print('found 2048')
            print(length)
        if length == 2048:
            #print('found 2048')
            print(length)
        i += 1
        

   

