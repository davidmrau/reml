import datasets
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
import evaluate


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


def exact_match_accuracy(references, predictions):
    return sum([any(p == r_ for r_ in r)for r, p in zip(references, predictions)])/len(references)

class Metrics:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def compute(self, predictions, references):
        if self.dataset_name == "nq_open":
            return {"EM": exact_match_accuracy(references, predictions)}
        else:
            raise KeyError()

