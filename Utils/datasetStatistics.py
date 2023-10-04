import argparse
import json
import pandas as pd
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from LLM.Scripts.evaluation_DocVQA import parse_args
from utils import build_dataset, load_config



def get_split_statistics(data, split):
    statistics = {}
    ids = []
    heights = []
    widths = []
    BB_heights = []
    BB_widths = []
    wrongOCR = []

    for item in tqdm.tqdm(data.dataset.imdb):

        if item['image_id'] not in statistics:
            
            ids.append(item['image_id'])
            statistics_doc = {}
            statistics_doc['img_height'] = item['image_height']
            statistics_doc['img_width'] = item['image_width']
            heights.append(item['image_height'])
            widths.append(item['image_width'])


            BB_height = 0
            BB_width = 0

            #get the mean height and width of the boxes
            for box in item['ocr_info']:
               
                BB_width += box['bounding_box']['width']*item['image_width']
                BB_height += box['bounding_box']['height']*item['image_height']
                
            if len(item['ocr_info']) == 0:
                statistics_doc['BB_mean_height'] = 0
                statistics_doc['BB_mean_width'] = 0
                BB_heights.append(0)
                BB_widths.append(0)
                wrongOCR.append(item['image_id'])
            else:
                    
                statistics_doc['BB_mean_height'] = BB_height/len(item['ocr_info'])
                statistics_doc['BB_mean_width'] = BB_width/len(item['ocr_info'])
                BB_heights.append(BB_height/len(item['ocr_info']))
                BB_widths.append(BB_width/(len(item['ocr_info'])))
                
            statistics[item['image_id']] = statistics_doc

    
    # create a pandas dataframe with the overall statistics statistics columns: statistics, min, max, average, std
    pandas_df = pd.DataFrame()

    #statistics: img_height, img_width, BB_mean_height, BB_mean_width
    pandas_df['statistics'] = ['img_height', 'img_width', 'BB_mean_height', 'BB_mean_width']
    pandas_df['min'] = [min(heights), min(widths), min(BB_heights), min(BB_widths)]
    pandas_df['max'] = [max(heights), max(widths), max(BB_heights), max(BB_widths)]
    pandas_df['average'] = [sum(heights)/len(heights), sum(widths)/len(widths), sum(BB_heights)/len(BB_heights), sum(BB_widths)/len(BB_widths)]
    pandas_df['std'] = [np.std(heights), np.std(widths), np.std(BB_heights), np.std(BB_widths)]

    print("max height: {} found in the document {}", max(heights), ids[np.argmax(heights)] )
    print("max width: {} found in the document {}", max(widths), ids[np.argmax(widths)] )


    #save the dataframe in a csv file
    pandas_df.to_csv('statistics_'+split+'.csv', index=False)
 
    return statistics, wrongOCR




if __name__ == "__main__":



    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='split to use')
    args = parser.parse_args()

    #load npy file
    config = load_config(parse_args())
    dataset = build_dataset(config, args.split)
    data = DataLoader(dataset, collate_fn=singlepage_docvqa_collate_fn)



    statistics, wrongOCR = get_split_statistics(data, args.split)
    print("The following statistics have been found for the split: ", args.split)
    print("A total of ", len(statistics), " documents have been found in the imdb")
    print("The following documents have been found with no OCR in the imdb: ", wrongOCR)
    

   

