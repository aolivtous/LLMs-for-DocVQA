import json
import pandas as pd

#read the data
data = json.load(open("/home/aolivera/TFM-LLM/LLM/Results/evaluation/PretaskT5_CLIP_unfrozen2023-07-31_14-52-59.json", 'r'))

# save the word and its score in a dataframe
df = pd.DataFrame(columns=['image_id', 'gt','pred', 'anls'])

for d in data['Scores by samples']:
    if data['Scores by samples'][d]['anls'] >= 0:
        df.loc[len(df)] = {'image_id': d, 'gt': "'"+ data['Scores by samples'][d]['gt_answer'][0], 'pred': "'"+ data['Scores by samples'][d]['pred_answer'][0], 'anls': data['Scores by samples'][d]['anls']} #the ' is added so that then we can open it with excel without problems with " at the beggining of the words"


# save the dataframe in a csv file
df.to_csv('/home/aolivera/TFM-LLM/LLM/Results/evaluation/PretaskT5_CLIP_unfrozen_allAnswers.csv', index=False,  sep='Ç')