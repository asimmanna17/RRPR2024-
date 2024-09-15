import math
import numpy as np
def nDCG(relevance_list, ideal_relevance_list):
    p = len(relevance_list)
    denominator = 1 / (np.log2(np.arange(p) + 2))
    idcg = np.sum([2**x - 1 for x in ideal_relevance_list] * denominator)
    dcg = np.sum([2**x - 1 for x in relevance_list] * denominator)
    # Normalize the cumulative gain by the ideal cumulative gain.
    if idcg==0:
        return 0
    else:
        ndcg = dcg / idcg
        return ndcg


def aCG(relevance_list):

    # Calculate the cumulative gain of the ranked list.
    cumulative_gain = 0.0
    for relevance in relevance_list:
        cumulative_gain += relevance

    # Calculate the average cumulative gain.
    return cumulative_gain / len(relevance_list)


def mAPw(relevance_list):
    # Assign weights based on relevance
    indicator = [1 if rel > 0 else 0 for rel in relevance_list]
    num_relevant_items = np.sum(indicator)


    # Calculate the average precision of the ranked list
    preclist = [0]*len(relevance_list)
    cumulative_relevance = 0.0
    for i, relevance in enumerate(relevance_list):
        #if relevance > 0:
        cumulative_relevance += relevance
        preclist[i]  = cumulative_relevance / (i + 1)

    if num_relevant_items ==0:
        weighted_mean_ap =0
    else:
        weighted_mean_ap = np.dot(relevance_list, preclist)/num_relevant_items
    return weighted_mean_ap

