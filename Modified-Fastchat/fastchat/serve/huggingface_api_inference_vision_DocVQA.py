import argparse
import json
import os
import numpy as np
import transformers
from transformers import CLIPImageProcessor 
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AddedToken
import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/model')
from docvqa_llm import DocVQALLM

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    other_tokens,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    for new_token in other_tokens:
        num_new_tokens += tokenizer.add_tokens(AddedToken(new_token, normalized=False))

    # model.resize_token_embeddings(len(tokenizer))
    model.llm_model.resize_token_embeddings(len(tokenizer))
    

    if num_new_tokens > 0:
        input_embeddings = model.llm_model.get_input_embeddings().weight.data
        output_embeddings = model.llm_model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    num_data: int = -1
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@torch.inference_mode()
def main(args, data, model, tokenizer):

    first_text = data['conversations'][0]['value'][0]
    second_text = data['conversations'][0]['value'][1]

    #check if first text ends with a space 
    if first_text[-1] != ' ':
        first_text += ' '
    
    #check if second text starts with a space
    if second_text[0] == ' ':
        second_text = second_text[1:]

    
    first_input_ids = tokenizer(["### Human: " + first_text]).input_ids 
    # remove last token (eos) and ensure it ends with 32106
    first_input_ids = [first_input_ids[0][:-1]]
    if first_input_ids[0][-1] != 32106:
        first_input_ids[0].append(32106)
    
    second_input_ids = tokenizer([second_text + "\n### Assistant:  </s>"]).input_ids
    #check if second text starts with 32106
    if second_input_ids[0][0] != 32106:
        second_input_ids[0].insert(0, 32106)

    input_ids1 = torch.tensor(first_input_ids).cuda()            
    first_input_embeds = model.llm_model.shared(input_ids1)

    input_ids2 = torch.tensor(second_input_ids).cuda()
    second_input_embeds = model.llm_model.shared(input_ids2)

    indeces = np.arange(int(data['start_idx']), int(data['end_idx'])+1)
    img_embeds = None
    space_embed =  model.llm_model.shared(torch.tensor([[32106]]).cuda())

    for idx in indeces:
        #load img embeds
        embed_file = os.path.join(args.embedsDir, args.split, data['document'] + '_' + str(idx)+".pt")
        img_embed = torch.load(embed_file).cuda()

        if args.clipOnly:
            img_embed = model.mm_projector(img_embed)

        if img_embeds is None:
            img_embeds = img_embed
        else:
            img_embeds = torch.cat([img_embeds,space_embed, img_embed], dim=1)


    to_regress_embeds = torch.cat([first_input_embeds,img_embeds, second_input_embeds], dim=1)
    
    output_ids = model.llm_model.generate(
        None,
        inputs_embeds=to_regress_embeds,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=60,
    )

    if model.llm_model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    print("Pred: "+ str(output_ids))
    return outputs
     


if __name__ == "__main__":

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(data_args.data_path) as f:
        data = json.load(f)


    # Add extra arguments
    parser.add_argument('--embedsDir', type=str, default='/data/users/aolivera/preprocess_CLIP/with_T5_custom', help='path where there are the embeddings and the json file')
    parser.add_argument('--modelPath', type=str, default='/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_pretask_CLIP_T5w_new/checkpoint-30000', help='path to the original T5 model weights')
    parser.add_argument('--split', type=str, default='val', help='split to use')
    parser.add_argument('--clipOnly', type=bool, default=True, help='Whether the embeddings are only from CLIP  or CLIP + linear layer')
    args = parser.parse_args()

    model = DocVQALLM.from_pretrained(args.modelPath,freeze_linear=True, vision_tower_type="CLIP",freeze_visionTower=True, freeze_llm=True)

    model = model.move_to_cuda()
    
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        'google/flan-t5-xl',
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        other_tokens=["<", "{", "\n", "}", "`", " ", "\\", "^", "\t"],
        tokenizer=tokenizer,
        model=model,
    )

    output_data = []

    for d in tqdm.tqdm(data):

        print("#############################################################")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("GT: "+ str(tokenizer.encode(d['conversations'][1]['value'])))
        outputs = main(args, d, model, tokenizer)

        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("GT: "+ d['conversations'][1]['value'])
        print("Pred: " + outputs.strip())

        d['conversations'][1]['value'] = outputs.strip()  
        output_data.append(d)

    #save the json file
    with open(training_args.output_dir, 'w') as f:
        json.dump(output_data, f)
    




