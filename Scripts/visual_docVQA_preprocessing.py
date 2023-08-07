import argparse
import json
import tqdm
import torch
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from evaluation_DocVQA import parse_args
from utils import build_dataset, load_config



def convert_json_to_new_format(normalizedData, validData,multiWords):
    new_data = []
    # add tqdm to show a progress bar

    #get all the valid questions id
    valid_questions_id = []
    for item in validData:
        valid_questions_id.append(int(item['id'].split('_')[1]))
    

    for item in tqdm.tqdm(normalizedData.dataset):


        if item['question_id'] in valid_questions_id:
            
            context_text = ' '.join(item['words'])
            context = "I have this document with the following words: " + context_text
            
            query =  context + '. ' + item['questions'] 
        
          
            answer = item['answers'][0] 

            if answer in query:

                answer_words = answer.split(' ')

                if multiWords:
                    word = answer
                else:
                    word = answer_words[torch.randint(0, len(answer_words), (1,)).item()]

                #check if answer is the start of the query
                if first_part_text == '':
                    print('answer is the start of the query')

                if second_part_text == '':
                    print('answer is the end of the query')

            else:
                #choose a random word from the context 
                context_words = item['words']
                word = context_words[torch.randint(0, len(context_words), (1,)).item()]
           
            first_part_text = query.split(word, 1)[0] #split only once
            second_part_text = query.split(word, 1)[1] 

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

            new_item = {
                'id': f'identity_{item["question_id"]}',
                'conversations': conversation
            }

            new_data.append(new_item)

            #process crop of the words to obtain the embeddings 
            #np.where(np.array(context_text) == word)[0]

            #/data/users/aolivera/preprocessCLIP/split/
    
    return new_data



if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='split to use')
    parser.add_argument('--output_file_withAnswers', type=str, default='val_visualData.json', help='path of the output JSON file that saves the context + questions in the FastChat format and the answers')
    parser.add_argument('--validQuestions', type=str, default='/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData.json', help='path of the config file')
    args = parser.parse_args()

    #load npy file
    config = load_config(parse_args())

    dataset = build_dataset(config, args.split)

    normalizedData = DataLoader(dataset, collate_fn=singlepage_docvqa_collate_fn)

    validData = None

    with open(args.validQuestions) as f:
        validData = json.load(f)
    

    if args.getTrain :    


        # Convert to new format
        new_format_data = convert_json_to_new_format(normalizedData,validData, isInference= False, withBB = args.withBB)


        # Convert to JSON string
        new_format_json_str = json.dumps(new_format_data, indent=2)

        # save the JSON string to a file

        with open(args.output_file_withAnswers, 'w') as f:
            f.write(new_format_json_str)

    

    inference_data = convert_json_to_new_format(normalizedData,validData, isInference= True, withBB = args.withBB)
    print("Number of valid questions ", len(inference_data))

    # Convert to JSON string
    inference_json_str = json.dumps(inference_data, indent=2)

    # save the JSON string to a file

    with open(args.output_file_forInference, 'w') as f:
        f.write(inference_json_str)

   

