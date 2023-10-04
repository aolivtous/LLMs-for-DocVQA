
import os
import random

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SPDocVQA(Dataset):

    def __init__(self, imbd_dir, images_dir, split, kwargs):
        data = np.load(os.path.join(imbd_dir, split, "imdb_{:s}.npy".format(split)), allow_pickle=True)
        self.header = data[0]
        self.imdb = data[1:]
        self.hierarchical_method = kwargs.get('hierarchical_method', False)

        self.max_answers = 2
        self.images_dir = images_dir

        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

    def __len__(self):
        return len(self.imdb)


    def __getitem__(self, idx):
        record = self.imdb[idx]
        question = record['question']
        context = ' '.join([word.lower() for word in record['ocr_tokens']])
        context_page_corresp = [0 for ix in range(len(context))]  # This is used to predict the answer page in MP-DocVQA. To keep it simple, use a mock list with corresponding page to 0.

        answers = list(set(answer.lower() for answer in record['answers']))         

        if self.get_raw_ocr_data:
            words = [word.lower() for word in record['ocr_tokens']]
            boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])

        start_idxs, end_idxs = self._get_start_end_idx(context, answers)

        sample_info = {'question_id': record['question_id'],
                       'questions': question,
                       'contexts': context,
                       'answers': answers,
                       'start_indxs': start_idxs,
                       'end_indxs': end_idxs
                       }
        
        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes
            sample_info['num_pages'] = 1
            sample_info['answer_page_idx'] = 0

        return sample_info

    def _get_start_end_idx(self, context, answers):

        answer_positions = []
        for answer in answers:
            start_idx = context.find(answer)

            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])

        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)  # If both answers are in the context. Choose one randomly.
        else:
            start_idx, end_idx = 0, 0  # If the indices are out of the sequence length they are ignored. Therefore, we set them as a very big number.

        return start_idx, end_idx


def singlepage_docvqa_collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch

