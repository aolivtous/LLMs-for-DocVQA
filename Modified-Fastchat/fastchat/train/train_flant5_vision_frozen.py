# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import defaultdict
import copy
import os
from dataclasses import dataclass, field
import random
import json
import logging
import pathlib
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import tqdm
import transformers
from transformers import CLIPImageProcessor 
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer, AddedToken
from transformers.models.t5 import modeling_t5

from fastchat.model.model_adapter import get_conversation_template

default_conversation = get_conversation_template("t5")

# TODO: import and use code from ../data/dataset.py

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/model')

from docvqa_llm_vision_frozen import DocVQALLM, smart_tokenizer_and_embedding_resize


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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    print("Saving model to {}".format(output_dir))
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        #cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()} #MODIFIED
        cpu_state_dict = {key: value for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn( 
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        images=None,
    )


def _form_qa(
    q_list,
    a_list,
    tokenized_conversation,
    tokenized_lens,
    speakers,
    header_len,
    max_length,
    eos_id,
):
    cur_idx = header_len
    conv_len = len(tokenized_conversation)

    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if cur_idx >= conv_len:
            break
        if speaker == "gpt":
            # truncate answer if it is too long
            content_a = None
            if tokenized_len > max_length:
                content_a = tokenized_conversation[cur_idx : cur_idx + max_length]
            else:
                content_a = tokenized_conversation[cur_idx : cur_idx + tokenized_len]
            content_a.append(eos_id)
            a_list.append(content_a)
            content_q = None
            if cur_idx >= max_length:
                content_q = tokenized_conversation[cur_idx - max_length : cur_idx]
            else:
                content_q = tokenized_conversation[:cur_idx]
            content_q.append(eos_id)
            q_list.append(content_q)
            # asser the last token is actually a EOS for an answer
            assert a_list[-1][-1] == eos_id, "Last Token is not EOS!"
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    
    conversation = header 
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": default_conversation.roles[0],  # human role
        "gpt": default_conversation.roles[1],  # gpt role
    }

    for i in range(len(source)):
        sentence = source[i]
        sentence_from = sentence["from"].lower()

        # TODO(Dacheng): verify this is a good way to split sentences
        if sentence_from == "human":
            # if this is not the last sentence
            if i != len(source) - 1:
                next_sentence = source[i + 1]
                sentence["value"] = (
                    BEGIN_SIGNAL
                    + roles.get(sentence_from, unknown_role)
                    + ": "
                    + sentence["value"][0] + sentence["value"][1] #MODIFIED
                    + END_SIGNAL
                    + BEGIN_SIGNAL
                    + roles.get(next_sentence["from"].lower(), unknown_role)
                    + ": "
                )
            else:
                # if human is the last speaker, it does not contribute to an answer
                pass
        else:
            sentence["value"] = sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    #print(conversation)
    return conversation


def preprocess(
    sources: Sequence[str],
    imgSources: Sequence[str],
    documents: Sequence[str],
    start_idx: Sequence[str],
    end_idx: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    tokenized_conversations = []
    tokenized_labels = []
    #header = f"{default_conversation.system}\n\n" # MODIFIED
    #header = "Be short, answer with one word if possible. "
    header = "" #MODIFIED
    for source in sources:
        #conversation = _add_speaker_and_signal(header, source, tokenizer)
        first_part = tokenizer("### Human: " + source[0]["value"][0],max_length=None, truncation=True)["input_ids"]

        if first_part[-1] == 1:
            first_part = first_part[:-1]
        if first_part[-1] != 32106:
            first_part = first_part + [32106]
        
        second_part = tokenizer(source[0]["value"][1] + "\n### Assistant: <\s>",max_length=None, truncation=True)["input_ids"]
        if second_part[0] != 32106:
            second_part = [32106] + second_part

        conversation = first_part + [-1] + second_part # To be able to split the conversation into two parts when we put the images in bewteen
        labels = tokenizer(source[1]["value"], max_length=None, truncation=True)["input_ids"]
        tokenized_labels.append(labels)
        tokenized_conversations.append(conversation)

    return dict(input_ids=tokenized_conversations, labels=tokenized_labels, embeds=imgSources, start_idx=start_idx, end_idx=end_idx, documents=documents)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        preprocessed_path,
        num_data,
       
    ):
        super(SupervisedDataset, self).__init__()
        
        # save to file
        # Make sure only the first process is processing the dataset
        #if dist.get_rank() != 0:
        #  dist.barrier()
        self.preprocessed_path = preprocessed_path
        if os.path.exists(self.preprocessed_path):
            logging.warning("loading from preprocessed data")
            with open(self.preprocessed_path, "r") as f:
                data_dict = json.load(f)
                
            #if dist.get_rank() == 0:
            #   dist.barrier()
        else:
            if not os.path.exists("preprocessed_data"):
                os.mkdir("preprocessed_data")
            #assert dist.get_rank() == 0, "Only the first process should process"
            logging.warning("Loading data...")
            list_data_dict = json.load(open(data_path, "r"))

            logging.warning("Formatting inputs...")
            sources = []

            sources = [example["conversations"] for example in list_data_dict]
            embeds = [example["embeds"] for example in list_data_dict]
            start_idx = [example["start_idx"] for example in list_data_dict]
            end_idx = [example["end_idx"] for example in list_data_dict]
            documents = [example["document"] for example in list_data_dict]

            data_dict = preprocess(sources, embeds, documents, start_idx, end_idx, tokenizer)
            json_data_dict = json.dumps(data_dict)


            # Remember to close file to avoid concurrent r/w
            with open(self.preprocessed_path,"w") as f: # what about the image tensors? we can't save them in the json 
                f.write(json_data_dict)

            # Release barrier
            #dist.barrier()

        if num_data != -1:
            data_dict["input_ids"] = data_dict["input_ids"][:num_data]
            data_dict["labels"] = data_dict["labels"][:num_data]
            data_dict["star_idx"] = data_dict["start_idx"][:num_data]
            data_dict["end_idx"] = data_dict["end_idx"][:num_data]
            data_dict["embeds"] = data_dict["embeds"][:num_data]
            data_dict["document"] = data_dict["document"][:num_data]

        full_data_dict = {}
        full_data_dict["input_ids"] = copy.deepcopy(data_dict["input_ids"])
        full_data_dict["labels"] = copy.deepcopy(data_dict["labels"])
        full_data_dict["images"] = self.load_image_embeds(data_dict["embeds"],data_dict["documents"],data_dict["start_idx"],data_dict["end_idx"])
        
        #import pdb; pdb.set_trace()

        # Shuffle data to see more conversations, if only train on partial data
        temp = list(zip(full_data_dict["input_ids"], full_data_dict["labels"], full_data_dict["images"]))
        random.shuffle(temp)
        res1, res2, res3= zip(*temp)
        full_data_dict["input_ids"], full_data_dict["labels"], full_data_dict["images"] = list(res1), list(res2), list(res3)


        # Dacheng: Get rid of short QA pair #MODIFIED
        self.input_ids = copy.deepcopy(full_data_dict["input_ids"])
        self.labels = copy.deepcopy(full_data_dict["labels"])
        self.images = copy.deepcopy(full_data_dict["images"])
        
    
    def __len__(self):
        return len(self.input_ids)

    
    def load_image_embeds(self, embed_sources, documents, start_idx, end_idx):
        
        embeddings = []
        # add tqdm
        
        for i, src in tqdm.tqdm(enumerate(embed_sources), total=len(embed_sources), desc="preprocessing embeds"):
            indices = list(range(int(start_idx[i]), int(end_idx[i])+1))
            
            embed = None
            for idx in indices:
                    path = os.path.join(src, documents[i]+ '_' + str(idx) + ".pt")
                    
                    if embed is None:
                        embed= torch.load(path)
                    else:
                        embed = torch.cat((embed, torch.load(path)), 1)

            embeddings.append(embed.cpu()) #because if not the Dataloader will crash in pin_memory method.... (torchdata/utils/_utils/pin_memory.py)
            
        return embeddings
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #import pdb; pdb.set_trace()

        return dict(input_ids=self.input_ids[i], images= self.images[i], labels=self.labels[i])





@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
   
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple(
            [
                torch.as_tensor(instance[key], dtype=torch.int64)
                for instance in instances
            ]
            for key in ("input_ids","labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        images_batch = [instance["images"] for instance in instances]

        ret = dict(
            input_ids=input_ids,
            images=images_batch,
            labels=labels,
        )
        torch.set_printoptions(profile="full")
        return ret



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        preprocessed_path=data_args.preprocessed_path,
        num_data=data_args.num_data,
        
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    val_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_path=data_args.data_path.replace("train", "val"),
        preprocessed_path=data_args.preprocessed_path.replace("train", "val"),
        num_data=data_args.num_data,
    )

    #print(data_args.num_data)
    return dict(
        train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator
    )

def train(from_pretrained, weights_path, vision_tower_type, freeze_linear, freeze_visionTower, freeze_llm):

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if from_pretrained:
        
        model = DocVQALLM.from_pretrained(weights_path, freeze_linear=freeze_linear, vision_tower_type=vision_tower_type, freeze_visionTower=freeze_visionTower, freeze_llm=freeze_llm)

    else:
        model = DocVQALLM(freeze_linear=freeze_linear, vision_tower_type=vision_tower_type, freeze_visionTower=freeze_visionTower, freeze_llm=freeze_llm)
    
    if freeze_linear:
        #check that the linear layer is frozen
        for param in model.mm_projector.parameters():
            if param.requires_grad:
                raise ValueError("Error The linear layer is not frozen")
    else:
        #check that the linear layer is not frozen
        for param in model.mm_projector.parameters():
            if not param.requires_grad:
                raise ValueError("Error The linear layer is frozen")
            
    if freeze_visionTower:
        #check that the vision tower is frozen
        for param in model.vision_tower.parameters():
            if param.requires_grad:
                raise ValueError("Error The vision tower is not frozen")
    else:
        #check that the vision tower is not frozen
        for param in model.vision_tower.parameters():
            if not param.requires_grad:
                raise ValueError("Error The vision tower is frozen")
    
    if freeze_llm:
        #check that the llm is frozen
        for param in model.llm_model.parameters():
            if param.requires_grad:
                raise ValueError("Error The llm is not frozen")
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

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
        model=model.llm_model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args) 
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Found existing checkpoint, resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No existing checkpoint, starting training from scratch")
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":

    #args.model_name_or_path = "google/flan-t5-xl" or the checkpoint of the T5 model we want to use (if from pretrained, it does not matter)

    from_pretrained = True
    full_model_weights_path = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_pretask_CLIP_T5w_new/checkpoint-30000"

    vision_tower_type = "CLIP"
    freeze_linear = False
    freeze_visionTower = True
    freeze_llm = True

    train(from_pretrained, full_model_weights_path, vision_tower_type, freeze_linear, freeze_visionTower, freeze_llm)