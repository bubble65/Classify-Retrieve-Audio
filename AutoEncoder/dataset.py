from torch.utils.data import Dataset
import os
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
class ESC50Dataset(Dataset):
    def __init__(self, data_dir, csv_file, mode, transform=None, label_mode="category"):
        """
        ESC-50 数据集加载与预处理
        Args:
            data_dir: 音频数据目录
            csv_file: 标签 CSV 文件路径
        """
        self.data_dir = data_dir
        self.transform = transform
        self.label_mode = label_mode
        # 读取标签文件
        self.data = []

        with open(csv_file, "r") as f:
            lines = f.readlines()[1:]  # 跳过标题行
            for line in lines:
                filename, fold, target = line.strip().split(",")[:3]
                # 5 折作为测试集
                if (fold == "5" and mode == "test") or (
                    fold != "5" and mode == "train"
                ):
                    self.data.append((filename, int(target)))

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        waveform, sr = torchaudio.load(os.path.join(self.data_dir, filename))
        # 转为 16kHz
        resampler = T.Resample(sr, 16000)  # [1 , 8w] 5s的音频
        waveform = resampler(waveform)
        # waveform = self.transform(waveform)
        if self.label_mode == 'tsne':
            label = label//10
        return waveform.squeeze(0), label

    def __len__(self):
        return len(self.data)

def get_data_loader(data_dir, csv_file, batch_size, mode, label_mode="category"):
    dataset = ESC50Dataset(data_dir, csv_file, mode, label_mode=label_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))