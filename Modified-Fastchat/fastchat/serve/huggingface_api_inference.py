"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from fastchat.utils import clean_flant5_ckpt
from fastchat.model import load_model, get_conversation_template, add_model_args


@torch.inference_mode()
def main(args, msg, model, tokenizer):


    if "vicuna" in args.model:
        #conv = get_conversation_template("vicuna")
        #conv = get_conversation_template(args.model)
        #conv.append_message(conv.roles[0], msg)
        #conv.append_message(conv.roles[1], None)
        #prompt = conv.get_prompt()

        # for inferecne (without fine-tune) we should add: Be short, answer with one word if possible. 
        input_ids = tokenizer(["Be short, answer with one word if possible USER: " + msg + " ASSISTANT: "]).input_ids #WITHOUT PUTING USER AND ASSISTANT IT DOES NOT WORK

    else: 
        # for inferecne (without fine-tune) we should add: Be short, answer with one word if possible. 
        input_ids = tokenizer(["Be short, answer with one word if possible ### Human: " + msg + " \n ### Assistant: "]).input_ids
    
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]

    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    #print(outputs)
    

    return outputs
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    #parser.add_argument("--model", type=str, default="/data/shared/vicuna/vicuna-7b")
    parser.add_argument("--model", type=str, default="lmsys/fastchat-t5-3b-v1.0")
    #parser.add_argument("--model", type=str, default="/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_latin_spaces_8epochs") 

    #parser.add_argument("--model", type=str, default="/home/aolivera/Documents/LLM_DocVQA/FastChat/checkpoints/checkpoints_T5_text/checkpoint-3300") 
    #parser.add_argument("--model", type=str, default="/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_8epochs/checkpoint-2700") 
   
    parser.add_argument("--json_file", type=str, default="/home/aolivera/TFM-LLM/LLM/Data/val_allQuestions_latin_spaces.json")
    parser.add_argument("--output_file", type=str, default="/home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_latin_spaces_8epochs.json")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    discarded_questions = []

    #clean the model when importing it from checkpoint for T5 
    #clean_flant5_ckpt(args.model)

    model, tokenizer = load_model(
        args.model, 
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
        outputs = main(args, msg, model, tokenizer)
        d['conversations'][1]['value'] = outputs.strip()
        valid_data.append(d)

    #save the json file
    with open(args.output_file, 'w') as f:
        json.dump(valid_data, f)
    




