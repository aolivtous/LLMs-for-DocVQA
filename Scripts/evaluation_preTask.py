import argparse
import json
import os, time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from SP_DocVQA import singlepage_docvqa_collate_fn
from metrics import EvaluatorPretask
from utils import load_config, save_json, build_dataset, extract_answers


def parse_args():
    parser = argparse.ArgumentParser(description='MP-DocVQA framework')
    parser.add_argument('--model', type=str, default = "vicuna7B", help='Name of the model')
    parser.add_argument('--predictedAnswers', type=str, default = "/home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_T5_CLIP_unfrozen_T5w.json", help='Path to predicted answers json file.')
    parser.add_argument('--gtAnswers', type=str, default = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced.json", help='Path to ground truth answers json file')
    parser.add_argument('--max-sequence-length', type=int, help='Max input sequence length of the model.')
    parser.add_argument('--save_dir', type=str, default = "/home/aolivera/TFM-LLM/LLM/Results/evaluation/" , help='path of the directory where the results folder will be saved')
    parser.add_argument('--method', type=str, default = "T5_CLIP_unfrozen_T5w", help='method used to generate the answers:')
    return parser.parse_args()

def evaluate(gt, evaluator, predAnswers, **kwargs):

    return_scores_by_sample = kwargs.get('return_scores_by_sample', False)
    return_answers = kwargs.get('return_answers', False)

    if return_scores_by_sample:
        scores_by_samples = {}
        total_anls = []
        total_ret_prec = []
        total_acc = 0

    else:
        total_anls = 0
        total_ret_prec = 0
        total_acc = 0

    all_pred_answers = []

    for gt_id in tqdm(gt):

        #if id exists (note that we had to filer some documents in the dataset because they were too long)
        if gt_id in predAnswers.keys():
            pred_answers = predAnswers[gt_id]
            
            metric = evaluator.get_metrics(gt[gt_id], pred_answers)

            if return_scores_by_sample:
                
                scores_by_samples[gt_id] = {
                    'anls': metric['anls'][0],
                    'pred_answer': pred_answers,
                    'gt_answer': gt[gt_id]
                }

            if return_scores_by_sample:
                total_anls.extend(metric['anls'])

            else:
                total_anls += sum(metric['anls'])
            
            if metric['anls'][0] == 1:
                total_acc += 1
              
            if return_answers:
                all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_anls = total_anls/len(gt)
        scores_by_samples = []
        total_acc = total_acc/len(gt)

    return total_anls,total_acc, total_ret_prec, all_pred_answers, scores_by_samples


if __name__ == '__main__':

    args = parse_args()
    start_time = time.time()

    evaluator = EvaluatorPretask(case_sensitive=False)
    
    predAnswers = json.load(open(args.predictedAnswers, 'r'))
    data = extract_answers(predAnswers)

    GT = json.load(open(args.gtAnswers, 'r'))
    gt_answers = extract_answers(GT)

    total_anls,total_acc, total_ret_prec, all_pred_answers, scores_by_samples = evaluate(gt_answers, evaluator, data, return_scores_by_sample=True, return_answers=True)
    anls = np.mean(total_anls)
    acc = total_acc/len(data)


    save_data = {
        "Method": args.method,
        "Number of questions": len(data),
        "Mean ANLS": anls,
        "Accuracy": acc,
        "Scores by samples": scores_by_samples,
    }

    experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(args.save_dir, "Pretask{:}{:}.json".format(args.method,experiment_date))
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))
