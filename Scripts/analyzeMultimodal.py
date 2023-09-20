import json
import pandas as pd

#read the data
data = json.load(open("/home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_vision_DocVQA_single_T5w.json", 'r'))
data2 = json.load(open("/home/aolivera/TFM-LLM/LLM/Results/evaluation/T5_val_visual_DocVQA_T5w_LINEAR_Allwords_ACC_2023-08-23_23-41-45.json", 'r'))
ids = {}
i = 0
for d in data:
    ids[d['id'].split('_')[1]] = i
    i += 1

# save the word and its score in a dataframe
df = pd.DataFrame(columns=['image_id', 'gt','pred', 'anls', 'fromAnswer', 'visualContext', 'lenAnswer', 'lenGT'])

for d in data2['Scores by samples']:

    lenAnswer = len(data2['Scores by samples'][d]['pred_answer'].split())
    lenGT = len(data2['Scores by samples'][d]['gt_answer'][0].split())


    if data2['Scores by samples'][d]['gt_answer'] == '':
        df.loc[len(df)] = {'image_id': d, 'gt': "'", 'pred': "'"+ data2['Scores by samples'][d]['pred_answer'], 'anls': data2['Scores by samples'][d]['anls'], 'fromAnswer': data[ids[d]]['fromAnswer'], 'visualContext': data[ids[d]]['visual_context'], 'lenAnswer': lenAnswer, 'lenGT': lenGT} #the ' is added so that then we can open it with excel without problems with " at the beggining of the words"
    else:
        df.loc[len(df)] = {'image_id': d, 'gt': "'"+ data2['Scores by samples'][d]['gt_answer'][0], 'pred': "'"+ data2['Scores by samples'][d]['pred_answer'], 'anls': data2['Scores by samples'][d]['anls'], 'fromAnswer': data[ids[d]]['fromAnswer'], 'visualContext': data[ids[d]]['visual_context'], 'lenAnswer': lenAnswer, 'lenGT': lenGT } #the ' is added so that then we can open it with excel without problems with " at the beggining of the words"
    
    
   

# save the dataframe in a csv file
df.to_csv('/home/aolivera/TFM-LLM/LLM/Results/evaluation/DocVQA_T5wLINEAR_1w.csv', index=False,  sep='Ç')