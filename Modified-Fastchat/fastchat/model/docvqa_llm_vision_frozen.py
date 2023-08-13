import logging
import os
import random
import numpy as np
from tokenizers import AddedToken

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import transformers
from typing import Dict, Optional, Sequence

from dataclasses import dataclass, field
import random
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
import logging

from transformers import CLIPVisionModel
from typing import Optional

from torchvision import models

from huggingface_hub import PyTorchModelHubMixin
from fastchat.model import load_model
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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


class DocVQALLM(nn.Module, PyTorchModelHubMixin):

    def __init__(
        self,
        freeze_linear=False,
        vision_tower_type="",
        freeze_visionTower=False,
        freeze_llm=True
        
    ):
        super().__init__()

        
        print('Loading visual encoder')

        if vision_tower_type == "CLIP":

            self.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_tower_inFeatures = self.vision_tower.config.hidden_size

            if freeze_visionTower:
                for name, param in self.vision_tower.named_parameters():
                    param.requires_grad = False
            else:
                for name, param in self.vision_tower.named_parameters():
                    param.requires_grad = True

            print('Loading projection layer')
        
        elif vision_tower_type == "ResNet":
            self.vision_tower = models.resnet18(pretrained=True)
            self.vision_tower_inFeatures = self.vision_tower.fc.in_features
            #remove the last layer
            
            self.vision_tower = nn.Sequential(*list(self.vision_tower.children())[:-1])

            if freeze_visionTower:
                for name, param in self.vision_tower.named_parameters():
                    param.requires_grad = False
            else:
                for name, param in self.vision_tower.named_parameters():
                    param.requires_grad = True
        

        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        print('Loading LLM')
        #check if the data path is empty, if so load the model from huggingface
       
        if model_args.model_name_or_path == "google/flan-t5-xl" :
            print('Loading T5 model from huggingface')
            self.llm_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir)
        
        else:
            print('Loading T5 model from local')
            self.llm_model, _ = load_model(model_args.model_name_or_path, "cuda" ,1)

        #freeze the llm model    
        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True


        if vision_tower_type == "CLIP" or vision_tower_type == "ResNet": 
            print('Loading projection layer')
            self.mm_projector = nn.Linear(self.vision_tower_inFeatures, self.llm_model.config.hidden_size)

            if freeze_linear:
                for name, param in self.mm_projector.named_parameters():
                    param.requires_grad = False
            else:
                for name, param in self.mm_projector.named_parameters():
                    param.requires_grad = True
                    
            #move the projection layer to the device
            self.mm_projector.to(self.llm_model.device)


        self.vision_tower.to(self.llm_model.device)
        print('Loading model done')

        
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            'google/flan-t5-xl',
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        
        smart_tokenizer_and_embedding_resize( 
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            other_tokens=["<", "{", "\n", "}", "`", " ", "\\", "^", "\t"],
            tokenizer=self.tokenizer,
            model=self.llm_model,
        )
    
    
    def forward(self, input_ids, images,labels):
        # load a tensor of images [batch_size, 3, 224, 224] for ResNet
        #images ja son els tensors
        fromCLIP = False
        fromCLIPLINEAR = False

        if images[0].size()[-1] == 768:
            fromCLIP = True
        elif images[0].size()[-1] == 2048:
            fromCLIPLINEAR = True
        
        img_embeds = []

        if fromCLIP:
            for i, img in enumerate(images):
                img_embeds.append(self.mm_projector(img))

        elif fromCLIPLINEAR:
            img_embeds = images
        else:
            print("Error: image size not supported, please use 768 if loading from CLIP or 2048 when loading from CLIP+LINEAR")

         
        space_embed =  self.llm_model.shared(torch.tensor([[32106]]).cuda())
        to_regress_embeds = []

        for j, batch_ids in enumerate(input_ids):

            if -1 in batch_ids:
                #split where the -1 is
                start_idx = torch.where(batch_ids == -1)[0][0].item()
                first_input_ids = batch_ids[:start_idx]
                second_input_ids = batch_ids[start_idx+1:]

                input_ids1 = torch.tensor(first_input_ids.unsqueeze(0))         
                first_input_embeds = self.llm_model.shared(input_ids1)

                input_ids2 = torch.tensor(second_input_ids.unsqueeze(0))
                second_input_embeds = self.llm_model.shared(input_ids2)

            else:
                print("Error: -1 not found in input_ids, there should always be a -1 in the input_ids to split the input_ids into two parts")
                print("Although all input contex may be visual, we will always have a first text saying: I have this document with the following words: and a second text containing the question ")

        
            result_shape = (1,img_embeds[j].shape[1]*2-1,img_embeds[j].shape[2])
            spaced_embeds = torch.zeros(result_shape)
            spaced_embeds = spaced_embeds.to('cuda')
        
            for i in range(img_embeds[j].shape[1]):
                start_idx = i*2
                spaced_embeds[:,start_idx,:] = img_embeds[j][:,i,:]
                
                if start_idx + 1 < result_shape[1]:
                    spaced_embeds[:,start_idx+1,:] = space_embed
            
            to_regress_embed = torch.cat([first_input_embeds,spaced_embeds, second_input_embeds], dim=1)
            to_regress_embeds.append(to_regress_embed)


        # padd the embeds to the max length
        max_dim = max([tensor.shape[1] for tensor in to_regress_embeds])

        padded_tensors = []
        for tensor in to_regress_embeds:
            padding = max_dim - tensor.shape[1]
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding, 0, 0))
            padded_tensors.append(padded_tensor)

        # concat the padded tensors along the first dimension to create the combined tensor [batch_size, max_dim, hidden_size]
        to_regress_pad_embeds = torch.concat(padded_tensors, dim=0)

        #extend all labels with -100 so that the eos token is not lost
        labels = torch.cat([labels, torch.full((labels.shape[0], 1), -100, dtype=torch.long).cuda()],  dim=1)
        
        if labels is not None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.llm_model._shift_right(labels)
        
        attention_mask = None

        encoded_tokens = self.llm_model.encoder(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=to_regress_pad_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        decoded_tokens = self.llm_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=encoded_tokens["last_hidden_state"],
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        #Added from modeling_t5.py T5ForConditionalGeneration

        lm_logits = self.llm_model.lm_head(decoded_tokens[0]) #same as (decoded_tokens['last_hidden_state']])
        #import pdb; pdb.set_trace()
        loss = None
        
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            print("###########")
            print("labels: ", labels[0])
            print("lm_logits: ", lm_logits.argmax(-1)[0])
            print("loss: ", loss.item())
           

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoded_tokens.past_key_values,
            decoder_hidden_states=decoded_tokens.hidden_states,
            decoder_attentions=decoded_tokens.attentions,
            cross_attentions=decoded_tokens.cross_attentions,
            encoder_last_hidden_state=encoded_tokens.last_hidden_state,
            encoder_hidden_states=encoded_tokens.hidden_states,
            encoder_attentions=encoded_tokens.attentions,
        )

    def gradient_checkpointing_enable(self): # TO DO #############
        return True # If I don't have this method the trainer.py will complain
    
    def encode_img_visionTower(self, images):

        if isinstance(self.vision_tower, nn.Sequential): # ResNet
            images = torch.stack(images)
            img_embeds = self.vision_tower(images)
            image_features = img_embeds.view(img_embeds.shape[0], -1)
        else: # CLIP
            image_features_list = []
            for image in images:
                # Cast image to float32
                image = image.type(torch.float32)
                img_embeds = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                img_embeds = img_embeds.hidden_states[-1][:, :1, :]
                image_features_list.append(img_embeds)

            image_features = torch.cat(image_features_list, dim=0)

        return image_features

    def move_to_cuda(self):

        self.llm_model.cuda()
        self.mm_projector.cuda()
        self.vision_tower.cuda()
        return self

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    other_tokens,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    ):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    for new_token in other_tokens:
        num_new_tokens += tokenizer.add_tokens(AddedToken(new_token, normalized=False))

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg