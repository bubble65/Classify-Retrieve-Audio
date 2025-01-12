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
        data.append(datas[i][0].cpu().detach().numpy())
        labels.append(datas[i][1].cpu().numpy())
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