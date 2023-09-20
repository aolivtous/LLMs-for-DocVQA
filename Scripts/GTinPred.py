import json
import pandas as pd

#read the data
data = json.load(open("/home/aolivera/TFM-LLM/LLM/Results/evaluation/T5_val_latin_spaces_2023-06-20_11-25-54.json", 'r'))
#/home/aolivera/TFM-LLM/LLM/Results/evaluation/T5_val_text_2023-06-06_22-28-14.json
#/home/aolivera/TFM-LLM/LLM/Results/evaluation/T5_val_latin_spaces_2023-06-20_11-25-54.json
#/home/aolivera/TFM-LLM/LLM/Results/evaluation/T5_val_BB_2023-06-06_22-28-35.json

GTINPRED = 0
for d in data['Scores by samples']:

    for gt in data['Scores by samples'][d]['gt_answer']:
        if gt in data['Scores by samples'][d]['pred_answer']:
            GTINPRED +=1

print("total gt in pred", GTINPRED/len(data['Scores by samples']))
   
