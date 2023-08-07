import editdistance


class Evaluator:
    def __init__(self, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        self.best_accuracy = 0
        # self.best_anls = 0
        self.best_epoch = 0

    def get_metrics(self, gt_answers, preds, update_global_metrics=True):
        
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])         
            batch_anls.append(self._calculate_anls(gt, pred))


        return {'anls': batch_anls}



    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()


    def _calculate_anls(self, gt, pred):
        if len(pred) == 0 or pred == 'none':
            return 0
        
        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls
    
    
    

class EvaluatorPretask:
    def __init__(self, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        self.best_accuracy = 0
        # self.best_anls = 0
        self.best_epoch = 0

    def get_metrics(self, gt_answers, preds, update_global_metrics=True):
        
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])

            batch_anls.append(self._calculate_anls(gt, pred))


        return {'anls': batch_anls}



    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()

        return string.strip()


    def _calculate_anls(self, gt, pred):
        if len(pred) == 0 or pred == 'none':
            return 0
        
        answers_similarity = 1 - self.get_edit_distance(gt, pred) / max(len(gt), len(pred)) 

        anls = answers_similarity if answers_similarity >= self.anls_threshold else 0
        return anls

    
    

if __name__ == '__main__':

    m = Evaluator()
    m.get_metrics(['aa', 'ab'], 'bb')
    print(m.get_metrics(['aa', 'ab'], 'bb'))