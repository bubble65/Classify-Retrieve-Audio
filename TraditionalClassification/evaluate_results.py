import json
import numpy as np
import os
import re


def extract_class(filename):
    """Extract class from filename."""
    pattern = re.compile(r"-(\d+).wav$")
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None


def calculate_binary_metric(results, top_k):
    """Calculate the binary metric: 1 if top_k contains the relevant class, else 0."""
    scores = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)
        if any(extract_class(file) == relevant_class for file in retrieved[:top_k]):
            scores.append(1)
        else:
            scores.append(0)
    return np.mean(scores)


def calculate_proportion_metric(results, top_k):
    """Calculate the proportion of top_k items that match the relevant class."""
    proportions = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)
        matches = sum(
            1 for file in retrieved[:top_k] if extract_class(file) == relevant_class
        )
        proportions.append(matches / top_k)
    return np.mean(proportions)


def calculate_mrr(results, top_k):
    """Calculate Mean Reciprocal Rank (MRR)."""
    reciprocal_ranks = []
    for query, retrieved in results.items():
        retrieved = retrieved[:top_k]
        relevant_class = extract_class(query)
        for rank, file in enumerate(retrieved, start=1):
            if extract_class(file) == relevant_class:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)


def calculate_ndcg(results, top_k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG)."""
    ndcgs = []
    for query, retrieved in results.items():
        relevant_class = extract_class(query)

        # Compute DCG
        dcg = 0
        for i, file in enumerate(retrieved[:top_k]):
            if extract_class(file) == relevant_class:
                dcg += 1 / np.log2(i + 2)  # i+2 because index starts at 0

        # Compute IDCG (ideal DCG)
        ideal_relevances = [1] * min(
            top_k,
            len([file for file in retrieved if extract_class(file) == relevant_class]),
        )
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))

        # Normalize DCG by IDCG
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)


def evaluate_metrics(result_file: str, top_k: list[int] = [10, 20]):
    """Evaluate MRR, NDCG, binary metric, and proportion metric from result file."""
    results = {}
    with open(result_file, "r") as f:
        for line in f:
            query_result = json.loads(line)
            results.update(query_result)

    ans = {}
    for k in top_k:
        mrr = calculate_mrr(results, k)
        ndcg = calculate_ndcg(results, k)
        binary_metric = calculate_binary_metric(results, k)
        proportion_metric = calculate_proportion_metric(results, k)

        ans.update(
            {
                f"MRR@{k}": mrr,
                f"NDCG@{k}": ndcg,
                f"Binary Metric@{k}": binary_metric,
                f"Proportion Metric@{k}": proportion_metric,
            }
        )
    return ans


def main(res_jsonl, top_k):
    metrics = evaluate_metrics(res_jsonl, top_k)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    # folder_path = "result"
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".jsonl"):
    #         print(filename)
    res_jsonl = "CLAP_embedding_20.jsonl"
    # filename = "CLAP_MLP_20.jsonl"
    # filename = "CLAP_zero_20.jsonl"
    # top_k = int(filename.split("_")[-1].split(".")[0])
    # metrics = evaluate_metrics(os.path.join(folder_path, filename), top_k)
    top_k = 10
    main(res_jsonl, top_k)
    # metrics = evaluate_metrics(filename, top_k)
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.4f}")
