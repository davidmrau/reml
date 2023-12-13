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


def exact_match_accuracy(predictions, references):
    """
    Compute exact match accuracy for a list of predictions and references.

    Args:
    - predictions (List[str]): List of predicted strings.
    - references (List[List[str]]): List of lists of reference strings (multiple references per prediction).

    Returns:
    - float: Exact match accuracy.
    """
    correct_count = 0
    total_count = len(predictions)

    for pred, refs in zip(predictions, references):
        # Check if the predicted string exactly matches any reference string
        if any(pred == ref for ref in refs):
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy

class Metrics(evaluate.Metric):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def _compute(self, predictions, references):
        if self.dataset_name == "nq_open":
            return {"EM": exact_match_accuracy(references, predictions)}
        else:
            raise KeyError()

