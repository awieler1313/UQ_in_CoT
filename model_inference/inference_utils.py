import numpy as np
from scipy.stats import beta
import math
from collections import Counter
import math
from collections import Counter
from sklearn.metrics import roc_auc_score

"""
This code is for creating the figures and doing the inference for the 
final paper, including figures and model performance
"""

def clopper_pearson_lower_bound(k, n, alpha):
    """
    Clopper Pearson lower bound for binomial proportion.
    k = number of successes
    n = number of trials
    """
    if k == n:
        # perfect precision, lower bound = 1
        return 1.0
    elif k == 0:
        # no successes, lower bound = 0
        return 0.0
    else:
        return beta.ppf(alpha, k, n - k + 1)

def clopper_pearson_interval(k, n, eps):
    """
    Two sided Clopper-Pearson confidence interval for binomial proportion.
    """
    if n == 0:
        return np.nan, np.nan

    if k == 0:
        lower = 0.0
        upper = beta.ppf(1 - eps/2, 1, n)
    elif k == n:
        lower = beta.ppf(eps/2, n, 1)
        upper = 1.0
    else:
        lower = beta.ppf(eps/2, k, n - k + 1)
        upper = beta.ppf(1 - eps/2, k + 1, n - k)

    return lower, upper


def find_cp_threshold(scores, labels, alpha=0.1, calibration_split=0.5, epsilon = 0.01):
    """
    find and return threshold t to guarantee precision according to 
    the threshold selection algorithm.
    """

    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]

    n1 = int(len(scores) * calibration_split)

    # split data based on calibration split value
    scores_A = scores[:n1]  
    labels_A = labels[:n1]

    scores_B = scores[n1:]  
    labels_B = labels[n1:]

    # sort candidate thresholds
    candidates = np.unique(scores_A)
    candidates.sort()

    best_t = None

    # Go through each tahreshold and check if it satisfies the condidiotns of the algorithm
    for t in candidates:
        mask = scores_B > t
        n = mask.sum()
        if n == 0:
            continue

        k = labels_B[mask].sum()

        # compute CP lower bound
        lower = clopper_pearson_lower_bound(k, n, epsilon)

        if lower >= 1 - alpha:
            best_t = t
            break

    return best_t, (scores_A, labels_A, scores_B, labels_B)



def evaluate_tensor(tensor_data, threshold=0.5, eps=1e-12):
    """
    Code to evaluate tensor of true vs predicted output values and return various scores
    """
    y_scores = tensor_data[:,0].cpu().numpy()
    y_true = tensor_data[:,1].cpu().numpy()

    bce = 0.0
    for yt, yp in zip(y_true, y_scores):
        yp = min(max(yp, eps), 1 - eps)  
        bce += -(yt * math.log(yp) + (1 - yt) * math.log(1 - yp))
    bce /= len(y_true)


    tp = sum(1 for yt, yp in zip(y_true, y_scores) if yt == 1 and yp >= threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_scores) if yt == 0 and yp >= threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_scores) if yt == 1 and yp < threshold)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = float('nan')

    return {
        "BCE": bce,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Total": len(y_true),
    }




def evaluate_baseline(dataset, threshold=0.5, eps=1e-12, method="soft_vote"):
    """
    Evaluate one of the two baseline models
    """
    y_true = []
    y_pred = []

    for item in dataset:
        responses = item["responses"]
        n = len(responses)

        if method == "hard_vote":
            # Most popular answer gets 1, others 0
            answers = [r["pred_answer"] for r in responses]
            counts = Counter(answers)
            most_common_count = counts.most_common(1)[0][1]

            # Check for tie
            if sum(1 for c in counts.values() if c == most_common_count) > 1:
                confidences = [0.0]*n
            else:
                most_common_answer = counts.most_common(1)[0][0]
                confidences = [1.0 if r["pred_answer"] == most_common_answer else 0.0 for r in responses]

        elif method == "soft_vote":
            # Confidence proportional to frequency
            answers = [r["pred_answer"] for r in responses]
            counts = Counter(answers)
            confidences = [counts[r["pred_answer"]] / n for r in responses]

        else:
            raise ValueError("Unknown method")

        for r, conf in zip(responses, confidences):
            y_true.append(r["correct"])
            y_pred.append(conf)

    bce = 0.0
    for yt, yp in zip(y_true, y_pred):
        yp = min(max(yp, eps), 1 - eps)
        bce += -(yt * math.log(yp) + (1 - yt)*math.log(1 - yp))
    bce/=len(y_true)

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp >= threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp >= threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp < threshold)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = float('nan')

    return {
        "BCE": bce,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Total": len(y_true),
    }

