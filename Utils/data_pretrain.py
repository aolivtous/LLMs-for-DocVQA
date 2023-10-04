import argparse
import json
import tqdm
from PIL import Image
import os


def get_preTrain_Data(data_dir, output_dir):
    new_data = []
    
    doc_dict = {}
    i = 0

    #iterate over the docunent folders not the files
    for document in tqdm.tqdm(os.listdir(data_dir)):
        if document.endswith('.txt'):
            continue
        if document.endswith('.json'):
            continue

        with open(os.path.join(data_dir,document, document + '.txt'), 'r') as f:
            #read the lines without the \n

            lines = [line.rstrip() for line in f]


            #iterate over the lines
            for j, line in enumerate(lines):

                conversation = [
                {
                    'from': 'human',
                    'value': "Type the following word: "
                },
                {
                    'from': 'gpt',
                    'value': line
                }
                ]

                new_item = {
                    'id': f'identity_{i}',
                    'image': os.path.join(data_dir,document, document + '_' + str(j) + '.png'),
                    'conversations': conversation
                }

                new_data.append(new_item)
                i += 1
    

        
    print("Total words = ", i)
    # save gt to json
    #with open(os.path.join(output_dir, 'pre_trainData.json'), 'w') as f:
    with open('pre_trainData_val.json', 'w') as f:
        json.dump(new_data, f)


    

if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='split to use')
    parser.add_argument('--data_dir', type=str, default='/home/aolivera/SinglePage_crops', help='path of the dataset')
    parser.add_argument('--output_dir', type=str, default='/home/aolivera/SinglePage_crops', help='path of to save the dataset')
    args = parser.parse_args()


    # Convert to new format
    get_preTrain_Data(os.path.join(args.data_dir, args.split), os.path.join(args.output_dir, args.split))

   

