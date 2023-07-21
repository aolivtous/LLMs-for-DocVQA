"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json
import transformers

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from fastchat.utils import clean_flant5_ckpt
from fastchat.model import load_model, get_conversation_template, add_model_args
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/model')

from docvqa_llm import DocVQALLM, smart_tokenizer_and_embedding_resize
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


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
def main(img, model, tokenizer):
    
    # for inferecne (without fine-tune) we should add: Be short, answer with one word if possible. 
    input_ids = tokenizer(["### Human: Type the following word: ### Assistant: "]).input_ids
    first_input_ids = input_ids[0:15]
    second_input_ids = input_ids[15:]

    img_embeds = DocVQALLM.encode_img(img) 

    first_input_embeds = model.shared(torch.tensor(first_input_ids).cuda())
    to_regress_second = model.shared(torch.tensor(second_input_ids).cuda())

    to_regress_embeds = torch.cat([first_input_embeds,img_embeds, to_regress_second], dim=1)



    output_ids = model.generate(
        None,
        inputs_embeds=to_regress_embeds,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=512,
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
   
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open(data_args.data_path) as f:
        data = json.load(f)


    model = DocVQALLM.load_pretrained_model(
                ckpt_path="/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_pretask_25_epochs_CLIP_pretrained/checkpoint-16500/rng_state.pth",
                freeze_linear=True,
                vision_tower_type="CLIP",
                freeze_visionTower=True,
                freeze_llm=True,

            )


    tokenizer = transformers.T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        other_tokens=["<", "{", "\n", "}", "`", " ", "\\", "^", "\t"],
        tokenizer=tokenizer,
        model=model.llm_model,
    )



    valid_data = []
    for d in tqdm.tqdm(data):

        msg =  d['conversations'][0]['value']
        outputs = main(msg, model, tokenizer)
        d['conversations'][1]['value'] = outputs.strip()
        valid_data.append(d)

    #save the json file
    #with open(args.output_file, 'w') as f:
    #    json.dump(valid_data, f)
    




