"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

from msclap.CLAPWrapper import CLAPWrapper
from esc50_dataset import ESC50,ESC50Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
# Load dataset
root_path = "../" # Folder with ESC-50-master/
dataset = ESC50(root=root_path, download=False) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
test_dataset = ESC50Dataset(root_path + 'ESC-50-master/audio/', root_path + 'ESC-50-master/meta/esc50.csv', 'test', label_mode='tsne')
train_dataset = ESC50Dataset(root_path + 'ESC-50-master/audio/', root_path + 'ESC-50-master/meta/esc50.csv', 'train')    
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False) 
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]


# Load and initialize CLAP
clap_model = CLAPWrapper(version = '2023', use_cuda=False)
text_embeddings = clap_model.get_text_embeddings(y)

# Computing text embeddings
#MLP
class MLP_M(nn.Module):   
    def __init__(self, input_dim, output_dim):
        super(MLP_M, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
best_acc = 0
model  = MLP_M(1024, 50)
state = torch.load('best_model.pth')
model.load_state_dict(state)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criteria = nn.CrossEntropyLoss()
for i in range(10):
    y_preds, y_labels = [], []
    model.train()
    _loss = 0
    for x,label in tqdm(train_dataloader):
        optimizer.zero_grad()
        audio_embeddings = clap_model.get_audio_embeddings(x, resample=True)
        similarity = model(audio_embeddings)
        loss = criteria(similarity, label)
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_pred = np.argmax(y_pred, axis=1)
        y_preds.append(y_pred)
        y_labels.append(label.cpu().numpy())
        _loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Loss {}'.format(loss.item()))
    print('Loss {}'.format(_loss/len(train_dataloader)))

    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(y_labels, y_preds)
    print('ESC50 Train Accuracy {}'.format(acc))

    y_preds, y_labels = [], []
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x,label in tqdm(test_dataloader):
            audio_embeddings = clap_model.get_audio_embeddings(x, resample=True)
            embeddings.append(audio_embeddings)
            similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
            y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_preds.append(y_pred)
            y_labels.append(label)

    embeddings = np.concatenate(embeddings, axis=0)
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    # print(y_labels[:10])
    # print(y_preds[:10])
    acc = accuracy_score(y_labels, y_preds)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
    print('ESC50 test Accuracy {}'.format(acc))
# y_preds, y_labels = [], []
# # model.eval()
# _loss = 0
# labels = []
# similaritys = []
# with torch.no_grad():
#     for i in tqdm(range(len(dataset))):
#         x,_,label = dataset.__getitem__(i)
#         filename = x.split('/')[-1] 
#         label = np.argmax(label)
#         audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
#         # similarity = model(audio_embeddings)
#         similaritys.append(audio_embeddings)
#         labels.append(filename)
#         # y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
#         # y_pred = np.argmax(y_pred, axis=1)
#         # y_preds.append(y_pred)
#         y_labels.append(label.cpu().numpy())

# result = {"similarities":similaritys, "labels":labels}
# torch.save(result, "CLAP_feature.pth")
# y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
# acc = accuracy_score(y_labels, y_preds)

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def visualize_tsne_2d(datas, perplexity=30, n_components=2, random_state=42, save_dir = 't-sne', modality = '', n_class = 10):
    """
    使用t-SNE进行降维可视化
    
    参数:
    data: numpy数组，形状为(n_samples, n_features)
    labels: 数据标签，可选
    perplexity: t-SNE的困惑度参数
    n_components: 降维后的维度
    random_state: 随机种子
    """
    data = []
    labels = []
    for i in range(len(datas)):
        data.append(datas[i][0])
        labels.append(datas[i][1])
    data = np.array(data)
    labels = np.array(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_class))
    custom_cmap = ListedColormap(colors)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state
    )
    tsne_results = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=labels,
            cmap=custom_cmap
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            alpha=0.5
        )
    
    plt.title(f't-SNE Visualization {modality} {n_components}d')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(save_dir)
    return plt.gcf()
output = []
for i in range(len(embeddings)):
    output.append((embeddings[i], y_labels[i]))
visualize_tsne_2d(output, save_dir='CLAP_t-sne.png', modality='audio', n_class=5)
"""
The output:

ESC50 Accuracy: 93.9%

"""