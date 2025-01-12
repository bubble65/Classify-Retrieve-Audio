import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from dataset import ESC50Dataset, get_data_loader
import logging
from features.AE import AutoEncoder, AE_trainer
from features.tsne import visualize_tsne_2d
from features.Another import extract_f0_using_cepstrum
import os
import argparse
# 配置日志

# 配置参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden_dim")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--output_model_path", type=str, default="./ckpt/AE_self/best_model.pth", help="output_model_path")
    parser.add_argument("--label_mode", type=str, default="tsne", help="label_mode")
    args = parser.parse_args()
    return args

@dataclass
class TrainArguments:
    """
    Arguments related to training.
    """

    data_dir: str = field(default="../../ESC-50-master/audio")
    csv_file: str = field(default="../../ESC-50-master/meta/esc50.csv")
    batch_size: int = field(default=16)
    learning_rate: float = field(default=1e-2)
    num_epochs: int = field(default=10)  # 差不多10轮就可以
    output_model_path: str = field(default="../ckpt/AE_self/best_model.pth")
    train_mode: bool = field(default=False)
    logger_path: str = field(default="./log/AE.log") 
    target_mode: str = field(default="fft")


@dataclass
class ModelArguments:
    """
    Arguments related to model initialization and configuration.
    """

    checkpoint_path: str = field(default="./ckpt/finetuned_model_1.pt")

if __name__ == '__main__':
    # 参数解析
    parser = HfArgumentParser((TrainArguments, ModelArguments))
    args = get_args()
    label_mode = args.label_mode
    train_args, model_args = parser.parse_args_into_dataclasses()

    # 加载数据
    train_dataloader =get_data_loader(train_args.data_dir, train_args.csv_file, train_args.batch_size, "train", label_mode="tsne")

    test_dataloader =get_data_loader(train_args.data_dir, train_args.csv_file, train_args.batch_size, "test", label_mode="tsne")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode
    hidden_dims = args.hidden_dim
    train_args.batch_size = args.batch_size
    train_args.num_epochs = args.epochs
    if mode == 'train':
        for hidden_dim in [hidden_dims]:
            # 设置logger路径
            path = train_args.logger_path.replace('.log', f'_{hidden_dim}.log')
            save_path = train_args.output_model_path.replace('.pth', f'_{hidden_dim}.pth')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, filename=path)
            logger = logging.getLogger(__name__)
            logger.info(f'hidden_dim: {hidden_dim}')
            print(f'hidden_dim: {hidden_dim}')
            model = AutoEncoder(16000 * 5, hidden_dim)
            trainer = AE_trainer(device=device, save_path=save_path, mask_ratio= 0.75, target_mode=None)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=train_args.learning_rate)
            trainer.fit(model, train_dataloader, test_dataloader, train_args.num_epochs, criterion, optimizer, logger)
            logger.info(f'hidden_dim: {hidden_dim} finished')
            print(f'hidden_dim: {hidden_dim} finished')
    if mode == 'test':
        hidden_dim = hidden_dims
        print(f'hidden_dim: {hidden_dim}')
        save_path = train_args.output_model_path.replace('.pth', f'_{hidden_dim}.pth')
        model = AutoEncoder(16000 * 5, hidden_dim)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        trainer = AE_trainer(device=device, save_path=save_path, mask_ratio= 0.75, target_mode=None)
        output, label = trainer.tsne(model, test_dataloader)
        outputs = []
        for i in range(len(output)):
            outputs.append((output[i], label[i]))
        visualize_tsne_2d(outputs, modality = 'AE', n_class = 5, save_dir="./ckpt/AE_self/AE_tsne.png", n_components=2)
    # count =1
    # for x,y in test_dataloader:
    #     for count in range(3):
    #         print(x[count].shape)
    #         print(y)
    #         f = extract_f0_using_cepstrum(x[count].numpy(), 16000)
    #         print(f)
    #     break

