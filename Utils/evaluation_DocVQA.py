import argparse
import json
import os, time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from SP_DocVQA import singlepage_docvqa_collate_fn
from metrics import Evaluator
from utils import load_config, save_json, build_dataset, extract_answers


def parse_args():
    parser = argparse.ArgumentParser(description='MP-DocVQA framework')
    parser.add_argument('--model', type=str, default = "T5", help='Name of the model')
    parser.add_argument('--predictedAnswers', type=str, default = "/home/aolivera/TFM-LLM/LLM/Results/inference/val_text_fastchat_8epoch.json", help='Path to predicted answers json file.')
    parser.add_argument('--dataset', type=str, default = "/home/aolivera/TFM-LLM/LLM/Configs/SP-DocVQA.yml", help='Path to yml file with dataset configuration.')
    parser.add_argument('--split', type=str, default = 'val', help='Dataset split: train, val, test.')
    parser.add_argument('--max-sequence-length', type=int, help='Max input sequence length of the model.')
    parser.add_argument('--save-dir', type=str, default = "/home/aolivera/TFM-LLM/LLM/Results/evaluation/" , help='path of the directory where the results folder will be saved')
    parser.add_argument('--context-type', type=str, default = "text_fastchat_8epoch", help='input context type for the model: text, textBB')
    #parser.add_argument('-bs', '--batch-size', type=int, help='DataLoader batch size.')
    
    #parser.add_argument('--seed', type=int, help='Seed to allow reproducibility.')
   
    return parser.parse_args()

def evaluate(data_loader, evaluator, predAnswers, **kwargs):

    return_answers = kwargs.get('return_answers', False)
    
    scores_by_samples = {}
    total_anls = []
    total_ret_prec = []
    total_gt_in_pred = []
    all_pred_answers = []
    gt_in_pred = 0
    total_acc = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch['question_id'])

        #if id exists (note that we had to filer some documents in the dataset because they were too long)
        if str(batch['question_id'][0]) in predAnswers.keys():
            pred_answers = predAnswers[str(batch['question_id'][0])]
            
            metric = evaluator.get_metrics(batch['answers'], pred_answers)

            for batch_idx in range(bs):

                if batch['answers'][batch_idx][0] in pred_answers[batch_idx]:
                    gt_in_pred = 1
                else:
                    gt_in_pred = 0
                
                total_gt_in_pred.append(gt_in_pred)

                scores_by_samples[batch['question_id'][batch_idx]] = {
                    'anls': metric['anls'][batch_idx],
                    'pred_answer': pred_answers[batch_idx],
                    'gt_answer': batch['answers'][batch_idx],
                    'gt_in_pred': gt_in_pred,
                }

            total_anls.extend(metric['anls'])

            if metric['anls'][0] == 1:
                total_acc += 1
            
            if return_answers:
                all_pred_answers.extend(pred_answers)


    return total_anls, total_gt_in_pred, total_acc, total_ret_prec, all_pred_answers, scores_by_samples


if __name__ == '__main__':

    args = parse_args()
    config = load_config(args)
    start_time = time.time()


    dataset = build_dataset(config, args.split)
    #val_data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=singlepage_docvqa_collate_fn)
    val_data_loader = DataLoader(dataset, collate_fn=singlepage_docvqa_collate_fn)

    evaluator = Evaluator(case_sensitive=False)
    
    predAnswers = json.load(open(args.predictedAnswers, 'r'))
    data = extract_answers(predAnswers)
    total_anls, total_gt_in_pred, total_acc, total_ret_prec, all_pred_answers, scores_by_samples = evaluate(val_data_loader, evaluator, data, return_answers=True)
    anls = np.mean(total_anls)
    acc = total_acc/len(data)
    gt_in_pred = np.sum(total_gt_in_pred)/len(total_gt_in_pred)


    save_data = {
        "Model": config["model"],
        #"Model_weights": config["model_weights"],
        "Dataset": config["dataset_name"],
        "Context type": args.context_type,
        "Number of questions": len(data),
        "Mean ANLS": anls,
        "GT in pred": gt_in_pred,
        "Accuracy": acc,
        "Scores by samples": scores_by_samples,
    }

    experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(config['save_dir'], "{:}_{:}_{:}_{:}.json".format(config['model'], args.split, args.context_type, experiment_date))
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))
