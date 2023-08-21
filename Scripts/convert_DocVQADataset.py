import argparse
import json
import tqdm
import torch
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from evaluation_DocVQA import parse_args
from utils import build_dataset, load_config

def get_context(words,boxes, withBB):
    #Append all words
    context = "I have this document with the following words: "
    for i,word in enumerate(words):
        if withBB:
            context += word + ":(" + str(int(boxes[i][0]*1000)) + ' ' + str(int(boxes[i][1]*1000)) + "), "
        else:
            context += word + " "
    return context

def convert_json_to_new_format(normalizedData, validData, isInference, withBB):
    new_data = []
    # add tqdm to show a progress bar

    #get all the valid questions id
    valid_questions_id = []
    for item in validData:
        valid_questions_id.append(int(item['id'].split('_')[1]))
    

    for item in tqdm.tqdm(normalizedData.dataset):


        if item['question_id'] in valid_questions_id:
        #if validData is None:
            context = get_context(item['words'],item['boxes'], withBB)
            query =  context + '. ' + item['questions'] 
        
            if isInference:
                answer = ''
            else :
                #answer = '. '.join(item['answers']) 
                answer = item['answers'][0] #only get the first answer

            conversation = [
                {
                    'from': 'human',
                    'value': query
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
    
    
    return new_data



if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='split to use')
    parser.add_argument('--getTrain', type=bool, default=False, help='get the train data')
    parser.add_argument('--output_file_withAnswers', type=str, default='test_validData.json', help='path of the output JSON file that saves the context + questions in the FastChat format and the answers')
    parser.add_argument('--output_file_forInference', type=str, default='test_validData_BB.json', help='path of the output JSON file that saves the context + questions in the FastChat ')
    parser.add_argument('--validQuestions', type=str, default='/home/aolivera/TFM-LLM/LLM/Scripts/test_validData_latin_spaces.json', help='path of the config file')
    parser.add_argument('--withBB', type=bool, default=True, help='use also the bounding boxes')
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

   

