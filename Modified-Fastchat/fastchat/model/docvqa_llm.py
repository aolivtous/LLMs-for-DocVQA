import logging
import random

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
from PIL import Image

from transformers import CLIPVisionModel
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from transformers import Trainer, AddedToken

from fastchat.model.model_adapter import get_conversation_template
from torchvision import models, transforms


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


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        
        self.fc1 = nn.Linear(64 * 56 * 56, 2048)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x) # [batch_size, 2048]

        x = x.unsqueeze(1) # [batch_size, 1, 2048]
        
        return x



class DocVQALLM(nn.Module):

    def __init__(
        self,
        freeze_linear=False,
        vision_tower_type="",
        freeze_visionTower=False,
        freeze_llm=True,
        
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        
        print('Loading visual encoder')
        #self.visual_encoder = CNNEncoder(embed_dim=768)

        if vision_tower_type == "CLIP":

            self.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_tower_inFeatures = self.vision_tower.config.hidden_size

            #device = "cuda" if torch.cuda.is_available() else "cpu"
            #self.vision_tower, preprocess = clip.load("ViT-B/32", device=device)
            #self.vision_tower.requires_grad_(False)
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

        else: # CNN

            self.vision_tower = CNNEncoder()

          
        """if pretrain_mm_mlp_adapter is not None:
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})"""



        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        print('Loading LLM')

        self.llm_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            model_args.model_name_or_path,
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

        #frize the llm model    
        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True


        if vision_tower_type == "CLIP" or vision_tower_type == "ResNet": 
            print('Loading projection layer')
            self.mm_projector = nn.Linear(self.vision_tower_inFeatures, self.llm_model.config.hidden_size)
            #move the projection layer to the device
            self.mm_projector.to(self.llm_model.device)
        
        #move vision tower to the device
        self.vision_tower.to(self.llm_model.device)
        print('Loading model done')
       


    def forward(self, input_ids, images,labels,attention_mask):
        # load a tensor of images [batch_size, 3, 224, 224] for ResNet
        
        img_embeds = self.encode_img(images) #What do we do with the attentions? -

        # remove the last token from the input_ids
        first_input_ids = input_ids[:, 0:15] # ### Human: Type the folowing word:
        second_input_ids = input_ids[:, 15:] # ### Assistant: </s>
        
        # remove eos token
        first_input_embeds = self.llm_model.shared(first_input_ids)
        second_input_embeds = self.llm_model.shared(second_input_ids)


        to_regress_embeds = torch.cat([first_input_embeds,img_embeds, second_input_embeds], dim=1) # correct size([batchsize=2, X, 2048])
    
        if labels is not None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.llm_model._shift_right(labels)
        
        # encode tokens 
        #encoded_tokens = self.llm_model.encoder(None, None, to_regress_embeds)
 
        # decode tokens
        #decoded_tokens = self.llm_model.decoder(None, None, encoded_tokens["last_hidden_state"])

        # lm head projection
        #outputs = self.llm_model.lm_head(decoded_tokens["last_hidden_state"])

        encoded_tokens = self.llm_model.encoder(
            input_ids=None,
            attention_mask=None,
            inputs_embeds=to_regress_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        # what about the attention mask????
        
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

        lm_logits = self.llm_model.lm_head(decoded_tokens["last_hidden_state"])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            #loss = loss_fct(lm_logits, labels)

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
    
    def encode_img(self, images):

        image_features = []
        
        if isinstance(self.vision_tower,CNNEncoder) or isinstance(self.vision_tower,nn.Sequential):
            images = torch.stack(images)
            
            img_embeds = self.vision_tower(images)
            image_features = self.mm_projector(img_embeds.view(img_embeds.shape[0], -1))
            image_features = image_features.unsqueeze(-2)
                
  
        
        else: # CLIP

            for image in images:

                img_embeds = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                img_embeds = img_embeds.hidden_states[-1][:, :1,:]
            
                image_features.append(img_embeds)

            image_features = torch.cat(image_features, dim=0)
            image_features = self.mm_projector(image_features)

        return image_features


    #@classmethod
    #def from_config(cls):
    def load_pretrained_model(ckpt_path, freeze_linear, vision_tower_type, freeze_visionTower,freeze_llm):

        model = DocVQALLM(
                freeze_linear=freeze_linear,
                vision_tower_type=vision_tower_type,
                freeze_visionTower=freeze_visionTower,
                freeze_llm=freeze_llm,
        )

        if ckpt_path:
            print("Load Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model

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
