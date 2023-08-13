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


def get_embeddings(args, question_id, indices, documentsInfo, model, image_processor):

    for idx in indices:
        img_path = os.path.join(args.cropedWords, args.split, documentsInfo[question_id], documentsInfo[question_id] + '_' + str(idx) + '.png')
        image = Image.open(img_path).resize((224, 224))
        img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = torch.stack([img])
        images = images.to('cuda')
        
        img_embeds = DocVQALLM.encode_img_visionTower(model,images)[0].cuda()

        if not args.clipOnly:
            img_embeds = model.mm_projector(img_embeds)

        img_embeds = img_embeds.unsqueeze(0) #batch size dim 1

        path =  os.path.join(args.output_dir, args.split, documentsInfo[question_id] + '_' + str(idx) + '.pt')

        #save the embeddings
        torch.save(img_embeds,path)
    
    

def find_word_number(words, start_index, end_index):
    
    current_index = 0
    
    for word_number, word in enumerate(words):
        current_index += len(word) + 1  # Add 1 for the space between words  
        
        if current_index > start_index:
            start_word_number = word_number
            break
    
    current_index = 0
    for word_number, word in enumerate(words):
        current_index += len(word) + 1  # Add 1 for the space between words
        
        if current_index > end_index:
            end_word_number = word_number 
            break
    
    return start_word_number, end_word_number

def check_OCR(docVQAdataset, validData):
    valid_ocr = {}
   
    #get all the valid questions id
    valid_questions_id = []
    for item in validData:
        valid_questions_id.append(int(item['id'].split('_')[1]))

    for item in tqdm.tqdm(docVQAdataset.dataset):

        if item['question_id'] in valid_questions_id:

            if item['start_indxs'] == 0 and item['end_indxs'] == 0:  
                valid_ocr[item['question_id']] = False

            else:
                valid_ocr[item['question_id']] = True
    
    
    answers_not_in_query = sum(1 for value in valid_ocr.values() if value is False)
    print('number of answers not from query: ', answers_not_in_query)
    return valid_ocr


def get_dataset_visualDocVQA(args, docVQAdataset, validData,documentsInfo, fromAnswer):
    new_data = []
  
    #get all the valid questions id
    valid_questions_id = []
    for item in validData:
        valid_questions_id.append(int(item['id'].split('_')[1]))

    
    model = DocVQALLM.from_pretrained(args.model, freeze_linear=True, vision_tower_type="CLIP",freeze_visionTower=True, freeze_llm=True)
    model = model.move_to_cuda()

    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    answer_not_in_query = 0

    for item in tqdm.tqdm(docVQAdataset.dataset):


        if item['question_id'] in valid_questions_id and fromAnswer[item['question_id']] != -1:
        
            prompt = "I have this document with the following words: "
            answer = item['answers'][0] 

            if args.fromAnswer == 1: #all from answer
                start_idx, end_idx = find_word_number(item['words'], item['start_indxs'], item['end_indxs'])

            else:
                if fromAnswer[item['question_id']] == 1: #from answer
                    start_idx, end_idx = find_word_number(item['words'], item['start_indxs'], item['end_indxs'])
                else: # choose another random word
                    answer_not_in_query += 1
                    if len(item['words']) == 0:
                        import pdb; pdb.set_trace()
                        
                    else: 
                        index = torch.randint(0, len(item['words']), (1,)).item()
                        visual_word = item['words'][index]
                        start_idx = end_idx = index 

    
            if args.multiWords:
                indices = np.arange(start_idx, end_idx+1)
                visual_word = ' '.join(item['words'][start_idx:end_idx+1])
                
            else: #choose a random word from the answer
                indices = [torch.randint(start_idx, end_idx+1, (1,)).item()]
                visual_word = item['words'][indices[0]]
                start_idx = end_idx = indices[0]


            first_part_words = item['words'][:start_idx]
            second_part_words = item['words'][end_idx+1:]

            first_part_text = prompt + ' '.join(first_part_words) 
            second_part_text = ' ' + ' '.join(second_part_words) + '. ' + item['questions']

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

            #get_embeddings(args, item['question_id'], indices, documentsInfo, model, image_processor)
            
            new_item = {
                'id': f'identity_{item["question_id"]}',
                'document': documentsInfo[item['question_id']],
                'embeds': os.path.join(args.output_dir, args.split),
                'start_idx': str(indices[0]),
                'end_idx': str(indices[-1]),
                'fromAnswer': fromAnswer[item['question_id']],
                'visual_context': visual_word,
                'conversations': conversation
            }

            new_data.append(new_item)

    if args.fromAnswer == 1:
        print('Valid data: ', len(new_data))
    else:
        print('number of answers not in query: ', answer_not_in_query)
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
    parser.add_argument('--model', type=str, default='/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_pretask_CLIP_T5w_new/checkpoint-30000', help='path to the model weights')
    #posar-lo en el .sh o en el launch
    parser.add_argument('--output_dir', type=str, default='/data/users/aolivera/preprocess_CLIP/with_T5_custom', help='path to save the embedding and the json file')
    parser.add_argument('--model_name_or_path', type=str, default='lmsys/fastchat-t5-3b-v1.0', help='path to the original T5 model weights')
    parser.add_argument('--multiWords', type=bool, default=False, help='replace all words of the answer by visual embeddings in multiwords answers')
    parser.add_argument('--onlyEmbeddings', type=bool, default=False, help='Compute and save the visual embeddings assuming that there is another json file is already created which selects which crops to use')
    parser.add_argument('--jsonData', type=str, default='/data/users/aolivera/preprocess_CLIP_LINEAR/with_T5_custom/val_visualDocVQA_singleWord_05.json', help='Path to the json file with the data')
    parser.add_argument('--fromAnswer', type=float, default=0.5, help='as per unit amount of times that the visual word is from the answer')
    parser.add_argument('--clipOnly', type=bool, default=True, help='Get the embeddings only with clip or with clip and the linear layer')
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
        
    
        #check which questions have answer in the OCR
        valid_OCR = check_OCR(docVQAdataset, validData)

        if args.fromAnswer == 1: #only questions with answer in the OCR
            from_answer = {key: (1 if value else -1) for key, value in valid_OCR.items()}

        else:
            # assign all values to 0
            from_answer = {key: 0 for key, value in valid_OCR.items()}

            # number of times that the visual word is from the answer
            ones = int(float(args.fromAnswer)*len(valid_OCR.items()))

            #create a list of the keys that have answer in the OCR, don't put the other keys in the list
            keys = [key for key, value in valid_OCR.items() if value]

            #shuffle keys
            random.shuffle(keys)

            if ones > len(keys):
                print("The number of times that the visual word is from the answer is too high, it will be set to the maximum possible value")
                ones = len(keys)
                print("The number of times that the visual word is from the answer is ", ones, " times")
                print("This corresponds to a ", str(ones/len(valid_OCR.items())*100), "% of the questions with answer in the OCR")
            
            #assign 1 to the first ones keys
            for key in keys[:ones]:
                from_answer[key] = 1
                    
        
        inference_data = get_dataset_visualDocVQA(args, docVQAdataset,validData,documentsInfo, from_answer)


        print("Number of valid questions ", len(inference_data))

        # Convert to JSON string
        inference_json_str = json.dumps(inference_data, indent=2)

    # save the JSON string to a file
    if args.multiWords:
        method = 'multiWords'
    else:
        method = 'singleWord'

    with open(os.path.join(args.output_dir, f"{args.split}_visualDocVQA_{method}_{str(args.fromAnswer).replace('.','')}.json"), 'w') as f:
        f.write(inference_json_str)

    

