import json
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
def main(img_path, model, tokenizer):

    #Preprocess the image
    image = Image.open(img_path).resize((224, 224))
  
    if isinstance(model.vision_tower,nn.Sequential): # ResNet
        
        image_rgb = image.convert('RGB') 

        normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.760289597495248,0.760289597495248,0.760289597495248], std=[0.3106214631171957,0.3106214631171957,0.3106214631171957])
            ])
        img = normalize(image_rgb)
    
    else: #CLIP 

        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    img.cuda()
                 

    if "toy5" in training_args.output_dir:
        #option5
        input_ids = tokenizer(["### Human: What does it say here: . Answer with just one word\n### Assistant:  </s>"]).input_ids 
        
        #input_ids= [[1713, 30345, 32106, 3892, 10, 32106, 363, 32106, 405, 32106, 34, 32106, 497, 32106, 270, 10, 32106, 3, 5, 32106, 11801, 32106, 28, 32106, 131, 32106, 80, 32106, 1448, 32103, 1713, 30345, 32106, 9255, 10, 32106, 1]]
        input_ids = torch.tensor(input_ids).cuda()
        input_embeds = model.llm_model.shared(input_ids)
        
        first_input_embeds = input_embeds[:, :17, :].cuda()
        second_input_embeds = input_embeds[:, 17:, :].cuda()
        
    
    else: #option1
        input_ids = tokenizer(["### Human: Type just the following text:  \n### Assistant:  </s>"]).input_ids 
        #input_ids = [[1713, 30345, 32106, 3892, 10, 32106, 6632, 32106, 131, 32106, 8, 32106, 826, 32106, 1499, 10, 32106, 32103, 1713, 30345, 32106, 9255, 10, 32106, 1]]
        input_ids = torch.tensor(input_ids).cuda()            
        input_embeds = model.llm_model.shared(input_ids)
        
        first_input_embeds = input_embeds[:, :17, :].cuda()
        second_input_embeds = input_embeds[:, 17:, :].cuda()

    #transform list of images to tensor
    images = torch.stack([img])
    images = images.to('cuda')
    
    img_embeds = DocVQALLM.encode_img(model,images)[0].cuda()
    img_embeds = img_embeds.unsqueeze(0) #batch size dim 1

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
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]

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


    model = DocVQALLM.from_pretrained("/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_pretask_10_epochs_CLIP_unfrozen_padToken/checkpoint-30000",freeze_linear=True, vision_tower_type="CLIP",freeze_visionTower=True, freeze_llm=True)
    #model2 = DocVQALLM(freeze_linear=True, vision_tower_type="CLIP",freeze_visionTower=True, freeze_llm=True)

    model = model.move_to_cuda()
    
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
        model=model,
    )

    output_data = []
    for d in tqdm.tqdm(data):
        print("#############################################################")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("GT: "+ str(tokenizer.encode(d['conversations'][1]['value'])))
        outputs = main(d['image'], model, tokenizer)

        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("GT: "+ d['conversations'][1]['value'])
        print("Pred: " + outputs.strip())

        d['conversations'][1]['value'] = outputs.strip()  
        output_data.append(d)

    #save the json file
    with open(training_args.output_dir, 'w') as f:
        json.dump(output_data, f)
    




