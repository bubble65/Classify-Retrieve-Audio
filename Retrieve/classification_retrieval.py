import torch
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

parameters = "CLAP_embedding.pth"
result = torch.load(parameters, weights_only=True)
filename = result["labels"]
weights = result["similarities"]
output_path = "CLAP_embedding.jsonl"

top_k = 20
train_set = []
test_set = []
for i in range(len(filename)):
    if filename[i].startswith("5-"):
        test_set.append({filename[i]: weights[i]})
    else:
        train_set.append({filename[i]: weights[i]})

for test in tqdm(test_set, desc="Retrieving"):
    score_list = []
    for train in train_set:
        test_embedding = list(test.values())[0].reshape(1, -1)
        train_embedding = list(train.values())[0].reshape(1, -1)
        score = cosine_similarity(test_embedding, train_embedding)[0, 0]
        score_list.append([score, list(train.keys())[0]])
    score_list.sort(key=lambda x: x[0], reverse=True)
    wav = [x[1] for x in score_list[:top_k]]
    new_dict = {"ESC-50-master/audio/" + list(test.keys())[0]: wav}
    with open(output_path, "a") as f:
        f.write(json.dumps(new_dict) + "\n")
