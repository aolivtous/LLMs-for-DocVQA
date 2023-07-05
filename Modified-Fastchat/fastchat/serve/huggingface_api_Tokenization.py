"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json
from os import path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from fastchat.model import load_model, add_model_args

@torch.inference_mode()
def check_tokenizer_length(msg, answer, tokenizer, max_length):
    import pdb; pdb.set_trace()
    if answer is not None:
        input_ids = tokenizer(["Be short, asnwer with one word if possible. \n### Human: " + msg + " \n### Assistant: " + answer + " "]).input_ids
    
    else: 
        input_ids = tokenizer(["\n### Human: " + msg + " \n### Assistant: "]).input_ids

    if len(input_ids[0]) < max_length and len(msg) != 0:
        return True
    else:
        #print('Discarded question: ', msg)
        return False
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, default="lmsys/fastchat-t5-3b-v1.0")
    parser.add_argument("--json_file", type=str, default="/home/aolivera/TFM-LLM/LLM/Data/old/val_AllData_BB.json")
    parser.add_argument("--output_file", type=str, default="/home/aolivera/TFM-LLM/LLM/Data/val_ValidData_BB.json")
    #parser.add_argument("--discarded_questions", type=str, default="/home/aolivera/Documents/LLM_DocVQA/data/train_discarded_questions.txt")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    discarded_questions = []

    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )

    valid_data = []
    for d in tqdm.tqdm(data):

        msg =  d['conversations'][0]['value']
        """if 'train' in args.json_file:
            answer = d['conversations'][1]['value']
        else:
            answer = None"""
        answer = d['conversations'][1]['value']

        valid_question = check_tokenizer_length(msg, answer, tokenizer, max_length=2048)
        if valid_question:
            valid_data.append(d)
        else:
            discarded_questions.append(d['id'])
 
        

    print('Valid questions: ', len(valid_data))
    print('Total questions: ', len(data))
    print('Discarded questions: ', str(len(data)-len(valid_data)))

    #save the json file
    with open(args.output_file, 'w') as f:
        json.dump(valid_data, f)
    
    # print the discarded questions
    print('Discarded questions: ', len(discarded_questions))


    """#save the discarded questions
    with open(args.discarded_questions, 'w') as f:
        for item in discarded_questions:
            f.write("%s\n" % item)"""


