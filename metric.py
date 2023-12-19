import datasets
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
import evaluate
import string
import regex 
import numpy as np

from evaluation.bem import BEM

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

def normalize(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def f1(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(predictions, references):
    return np.mean([max([f1(prediction, gt) for gt in ground_truths]) for ground_truths, predicition in zip(references, predicions)])

def em(prediction, ground_truth):
    print(prediction, ground_truth)
    return float(normalize(prediction) == normalize(ground_truth))


def exact_match_score(predictions, references):
    return np.mean([max([em(prediction, gt) for gt in ground_truths]) for ground_truths, prediction in zip(references, predictions)])



class Metrics:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.bem = BEM()

    def compute(self, predictions, references, questions=None):
        if self.dataset_name == "nq_open":
            return {
                        "EM": exact_match_score(predictions, references),
                        "BEM": self.bem(references, predictions, questions),
                        "f1": f1_score(predictions, references),
                    }
        else:
            raise KeyError()

