import argparse
import json
import os
import random
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from utils import build_dataset, load_config
from transformers import CLIPImageProcessor 
from PIL import Image
import transformers
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(1, '/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/model')
from docvqa_llm import DocVQALLM


def get_embeddings(args, question_id, indices_batch, documentsInfo, model, image_processor):
    img_batch = []
    for idx in indices_batch:
        img_path = os.path.join(args.cropedWords, args.split, documentsInfo[question_id], documentsInfo[question_id] + '_' + str(idx) + '.png')
        image = Image.open(img_path).resize((224, 224))
        img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        img_batch.append(img.to('cuda'))
            

    #img_batch = img_batch.to('cuda')
    
    img_embeds = DocVQALLM.encode_img_visionTower(model,img_batch).cuda()

    if not args.clipOnly:
        img_embeds = model.mm_projector(img_embeds)
    
    size = img_embeds.size()[-1]

    img_embeds_list = [tensor.view(1, 1, size) for tensor in img_embeds]

    for i,idx in enumerate(indices_batch):
        path =  os.path.join(args.output_dir, args.split, documentsInfo[question_id] + '_' + str(idx) + '.pt')
        torch.save(img_embeds_list[i],path)
    
   

def get_dataset_visualDocVQA_allContext(args, docVQAdataset, validData,documentsInfo):
    new_data = []
  
    #get all the valid questions id
    valid_questions_id = []
    for item in validData:
        valid_questions_id.append(int(item['id'].split('_')[1]))

    
    model = DocVQALLM.from_pretrained(args.model, freeze_linear=True, vision_tower_type="CLIP",freeze_visionTower=True, freeze_llm=True)
    model = model.move_to_cuda()

    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    i = 0

    for item in tqdm.tqdm(docVQAdataset.dataset):


        if item['question_id'] in valid_questions_id :
        
            first_part_text = "I have this document with the following words: "
            second_part_text = ' . ' + item['questions']

            answer = item['answers'][0] 
            start_idx = 0
            end_idx = len(item['words'])

            indices = np.arange(start_idx, end_idx)

            conversation = [
                {
                    'from': 'human',
                    'value': [first_part_text, second_part_text]
                },
                {
                    'from': 'gpt',
                    'value': answer
                }
            ]

            #split indices into batches
            num_batches = (len(indices) + args.batch_size - 1) // args.batch_size

            # Split indices into batches
            indices_array = [indices[i * args.batch_size: (i + 1) * args.batch_size] for i in range(num_batches)]
            for batch in indices_array:
                get_embeddings(args, item['question_id'], batch, documentsInfo, model, image_processor)
                
        
            new_item = {
                'id': f'identity_{item["question_id"]}',
                'document': documentsInfo[item['question_id']],
                'embeds': os.path.join(args.output_dir, args.split),
                'start_idx': str(indices[0]),
                'end_idx': str(indices[-1]),
                'fromAnswer': 1,
                'visual_context': item['words'],
                'conversations': conversation
            }
            
            new_data.append(new_item)
            i += 1

    

   
    print('Valid data: ', len(new_data))
    
    return new_data


if __name__ == "__main__":

    #arguments arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='split to use')
    parser.add_argument('--dataset', type=str, default='/home/aolivera/TFM-LLM/LLM/Configs/SP-DocVQA.yml', help='Path to yml file with dataset configuration.')
    parser.add_argument('--validQuestions', type=str, default='/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_validData.json', help='path of the config file')
    parser.add_argument('--documentsInfo', type=str, default='/home/aolivera/SinglePage_DocVQA/', help='path of the documents')
    parser.add_argument('--cropedWords', type=str, default='/home/aolivera/SinglePage_crops', help='path of the folder with the croped words')
    parser.add_argument('--model', type=str, default='/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_pretask_CLIP_unfrozen_T5g_new_weights/checkpoint-43750', help='path to the model weights')
    #posar-lo en el .sh o en el launch
    parser.add_argument('--output_dir', type=str, default='/data/users/aolivera/preprocess_CLIP_LINEAR/with_T5_fastchat', help='path to save the embedding and the json file')
    parser.add_argument('--model_name_or_path', type=str, default='lmsys/fastchat-t5-3b-v1.0', help='path to the original T5 model weights')
    parser.add_argument('--onlyEmbeddings', type=bool, default=False, help='Compute and save the visual embeddings assuming that there is another json file is already created which selects which crops to use')
    parser.add_argument('--jsonData', type=str, default='/data/users/aolivera/preprocess_CLIP/with_T5_custom/val_visualDocVQA_CLIP_singleWord_05.json', help='Path to the json file with the data')
    parser.add_argument('--clipOnly', type=bool, default=False, help='Get the embeddings only with clip or with clip and the linear layer')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size to compute the embeddings')
    args = parser.parse_args()


    #join documentsInfo with /split/split_v1.0.json
    joined_path = os.path.join(args.documentsInfo, args.split, f"{args.split}_v1.0.json")
    
    with open(joined_path) as f:
        documents = json.load(f)

    documentsInfo = {}
    for item in documents['data']:
        documentsInfo[item['questionId']] = item['ucsf_document_id'] + '_' + item['ucsf_document_page_no']


    if args.onlyEmbeddings:

        model = DocVQALLM.from_pretrained(args.model, freeze_linear=True, vision_tower_type="CLIP",freeze_visionTower=True, freeze_llm=True)
        model = model.move_to_cuda()
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        with open(args.jsonData) as f:
            data = json.load(f)

        for item in tqdm.tqdm(data):
            indices = np.arange(int(item['start_idx']), int(item['end_idx'])+1)
            paths = get_embeddings(args, int(item['id'].split("_")[1]), indices, documentsInfo, model, image_processor)
            item['embeds'] = paths

        inference_json_str = json.dumps(data, indent=2)
            


    else:
        #load npy file
        config = load_config(args)

        dataset = build_dataset(config, args.split)

        docVQAdataset = DataLoader(dataset, collate_fn=singlepage_docvqa_collate_fn)

        validData = None

        with open(args.validQuestions) as f:
            validData = json.load(f)
        
      
        inference_data = get_dataset_visualDocVQA_allContext(args, docVQAdataset,validData,documentsInfo)

        
        # Convert to JSON string
        inference_json_str = json.dumps(inference_data, indent=2)

    
    method = 'allContext'
   
    with open(os.path.join(args.output_dir, f"{args.split}_visualDocVQA_{method}_1.json"), 'w') as f:
        f.write(inference_json_str)

    

