import json
import numpy as np
import os
import re

def extract_class(filename):
    pattern = re.compile(r'-(\d+).wav$')
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None

def calculate_binary_metric(results, top_k):
    scores = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)
        if any(extract_class(file) == relevant_class for file in retrieved[:top_k]):
            scores.append(1)
        else:
            scores.append(0)
    return np.mean(scores)

def calculate_proportion_metric(results, top_k):
    proportions = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)
        matches = sum(1 for file in retrieved[:top_k] if extract_class(file) == relevant_class)
        proportions.append(matches / top_k)
    return np.mean(proportions)

def calculate_mrr(results, top_k):
    reciprocal_ranks = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)
        flag = False
        for rank, file in enumerate(retrieved[:top_k], start=1):
            if extract_class(file) == relevant_class:
                reciprocal_ranks.append(1 / rank)
                flag = True
                break
        if flag == False:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

def calculate_ndcg(results, top_k):
    ndcgs = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)
        
        dcg = 0
        for i, file in enumerate(retrieved[:top_k]):
            if extract_class(file) == relevant_class:
                dcg += 1 / np.log2(i + 2)
        
        ideal_relevances = [1] * min(top_k, len([file for file in retrieved if extract_class(file) == relevant_class]))
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
        
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)

def evaluate_metrics(result_file, top_k):
    results = {}
    with open(result_file, 'r') as f:
        for line in f:
            query_result = json.loads(line)
            results.update(query_result)

    mrr = calculate_mrr(results, top_k)
    ndcg = calculate_ndcg(results, top_k)
    binary_metric = calculate_binary_metric(results, top_k)
    proportion_metric = calculate_proportion_metric(results, top_k)

    return {
        "MRR": mrr,
        "NDCG": ndcg,
        "Binary Metric": binary_metric,
        "Proportion Metric": proportion_metric
    }

if __name__ == "__main__":
    folder_path = "result"
    for filename in os.listdir(folder_path):
        print(filename)
        for top_k in [10, 20]:
            metrics = evaluate_metrics(os.path.join(folder_path, filename), top_k)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        print()
