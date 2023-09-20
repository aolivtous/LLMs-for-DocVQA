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
    parser.add_argument('--predictedAnswers', type=str, default = "/home/aolivera/TFM-LLM/LLM/Results/inference/val_TEST_inference_vision_Pretask_T5_CLIP_unfrozen_T5_flant_w_New.json", help='Path to predicted answers json file.')
    parser.add_argument('--gtAnswers', type=str, default = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced_test_1000.json", help='Path to ground truth answers json file')
    parser.add_argument('--save_dir', type=str, default = "/home/aolivera/TFM-LLM/LLM/Results/evaluation/" , help='path of the directory where the results folder will be saved')
    parser.add_argument('--method', type=str, default = "T5_CLIP_unfrozen_T5w_TEST_new", help='method used to generate the answers that will serve as the output name of the results')
    return parser.parse_args()

def evaluate(gt, evaluator, predAnswers, **kwargs):

    return_answers = kwargs.get('return_answers', False)

    scores_by_samples = {}
    total_anls = []
    total_anls_first_word = []
    total_acc_first_word = 0
    total_ret_prec = []
    total_acc = 0
    total_gt_in_pred = 0
    gt_in_pred = 0

    all_pred_answers = []

    for gt_id in tqdm(gt):
        gt_in_pred = 0

        #if id exists (note that we had to filer some documents in the dataset because they were too long)
        if gt_id in predAnswers.keys():
            pred_answers = predAnswers[gt_id]
            metric = evaluator.get_metrics(gt[gt_id], pred_answers)
            metric_first_word = evaluator.get_metrics(gt[gt_id], [pred_answers[0].split(" ")[0]])
            for g in gt[gt_id]:
                if g in pred_answers[0]:
                    total_gt_in_pred += 1
                    gt_in_pred = 1
                    break
                else:
                    gt_in_pred = 0

            scores_by_samples[gt_id] = {
                'anls': metric['anls'][0],
                'anls_first_word': metric_first_word['anls'][0],
                'gt_in_pred': gt_in_pred,
                'pred_answer': pred_answers[0],
                'gt_answer': gt[gt_id],
                
            }

            total_anls.extend(metric['anls'])
            total_anls_first_word.extend(metric_first_word['anls'])

            if metric['anls'][0] == 1:
                total_acc += 1
            if metric_first_word['anls'][0] == 1:
                total_acc_first_word += 1
              
            if return_answers:
                all_pred_answers.extend(pred_answers)

    return total_anls,total_anls_first_word, total_acc, total_acc_first_word, total_gt_in_pred, total_ret_prec, all_pred_answers, scores_by_samples


if __name__ == '__main__':

    args = parse_args()
    start_time = time.time()

    evaluator = EvaluatorPretask(case_sensitive=False)
    
    predAnswers = json.load(open(args.predictedAnswers, 'r'))
    data = extract_answers(predAnswers)

    GT = json.load(open(args.gtAnswers, 'r'))
    gt_answers = extract_answers(GT)

    total_anls,total_anls_first_word, total_acc, total_acc_first_word, total_gt_in_pred, total_ret_prec, all_pred_answers, scores_by_samples = evaluate(gt_answers, evaluator, data, return_answers=True)
    anls = np.mean(total_anls)
    anls_first_word = np.mean(total_anls_first_word)
    acc = total_acc/len(data)
    acc_first_word = total_acc_first_word/len(data)
    gt_in_pred = total_gt_in_pred/len(data)


    save_data = {
        "Method": args.method,
        "Number of questions": len(data),
        "Mean ANLS": anls,
        "Mean ANLS first word": anls_first_word,
        "Accuracy": acc,
        "Accuracy first word": acc_first_word,
        "GT in pred": gt_in_pred,
        "Scores by samples": scores_by_samples,
    }

    experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(args.save_dir, "Pretask{:}{:}.json".format(args.method,experiment_date))
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))
