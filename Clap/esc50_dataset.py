from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import torch.nn as nn
import torch
import torchaudio
import torchaudio.transforms as T
class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        # if download:
        #     self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class2idx = {'Dog': 1, 'Rooster': 2, 'Pig': 3, 'cow': 4, 'Frog': 5, 'Cat': 6, 'Hen': 7, 'Insects': 8, 'Sheep': 9, 'Crow': 10, 'Rain': 11, 'Sea waves': 12, 'Crackling fire': 13, 'Crickets': 14, 'Chirping birds': 15, 'Water drops': 16, 'wind': 17, 'Pouring water': 18, 'Toilet flush': 19, 'Thunderstorm': 20, 'Crying baby': 21, 'Sneezing': 22, 'Clapping': 23, 'Breathing': 24, 'Coughing': 25, 'Footsteps': 26, 'Laughing': 27, 'Brushing teeth': 28, 'Snoring': 29, 'Drinking sipping': 30, 'Door wood knock': 31, 'Mouse click': 32, 'Keyboard typing': 33, 'Door wood creaks': 34, 'Can opening': 35, 'Washing machine': 36, 'Vacuum cleaner': 37, 'Clock alarm': 38, 'Clock tick': 39, 'Glass breaking': 40, 'Helicopter': 41, 'Chainsaw': 42, 'Siren': 43, 'Car horn': 44, 'Engine': 45, 'Train': 46, 'Church bells': 47, 'Airplane': 48, 'Fireworks': 49, 'Hand saw': 50}
for key in list(class2idx.keys()):
    class2idx[key.lower()] = class2idx[key]
class ESC50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta','esc50.csv'),
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = False):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        # self.df['filename'] = os.path.join(self.root, self.base_folder, self.audio_dir) + os.sep + self.df['filename']
        self.df['category'] = self.df['category'].str.replace('_',' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_',' ') for x in self.df[self.label_col].unique()]
        for i, category in enumerate(self.classes):
            print(f"category: {category} ----> {class2idx[category.lower()] -1 }")
            self.class_to_idx[category] = class2idx[category.lower()] -1
        self.classes = sorted(self.classes, key=lambda x: self.class_to_idx[x])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1,-1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        # Download file using requests
        import requests
        file = Path(self.root) / self.filename
        if file.is_file():
            return
        
        r = requests.get(self.url, stream=True)

        # To prevent partial downloads, download to a temp file first
        tmp = file.with_suffix('.tmp')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'wb') as f:
            pbar = tqdm(unit=" MB", bar_format=f'{file.name}: {{rate_noinv_fmt}}')

            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk) / 1024 / 1024)
                    f.write(chunk)
                    
        # move temp file to correct location
        tmp.rename(file)
        
        # # extract file
        from zipfile import ZipFile
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)

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
        # 转为 16kHz
        filename = os.path.join(self.data_dir, filename)
        if self.label_mode == 'tsne':
            label = label//10
        return filename, label

    def __len__(self):
        return len(self.data)
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_',' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i
