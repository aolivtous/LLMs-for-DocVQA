import argparse
import json
from LLM.Scripts.evaluation_DocVQA import parse_args
from utils import load_config


if __name__ == "__main__":

    input_file = ""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    new_format = []
    for item in data:
        new_data = {}
        new_data['answer'] = item['conversations'][1]['value']
        new_data['questionId'] = int(item['id'].split('_')[1])
        new_format.append(new_data)
    
    with open(input_file.split('.')[0] + '_submissionFormat.json', 'w') as f:
        json.dump(new_format, f)
        

   