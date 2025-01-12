import torch
import torch.nn as nn
import os
import tqdm
from numpy.fft import fft
import numpy as np
import librosa
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = None):
        super(AutoEncoder, self).__init__()
        if output_size is None:
            output_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, output_size) 
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def extract_feature(self, x):
        return self.encoder(x)
    
class AE_trainer(nn.Module):
    def __init__(self,device, save_path, mask_ratio=0, target_mode='fft'):
        super(AE_trainer, self).__init__()
        self.device = device
        self.best = 1e10
        self.best_epoch = 0
        self.save_path = save_path
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.mask_ratio = mask_ratio
        self.target_mode = target_mode
    def fit(self, model ,train_dataloader, test_dataloader, num_epochs, criterion, optimizer, logger):
        for epoch in range(num_epochs):
            model.train()
            self.train(model, train_dataloader, criterion, optimizer, epoch, logger)
            model.eval()
            self.test(model, test_dataloader, criterion, logger, epoch)
        print(f'Best Loss on Epoch {self.best_epoch}: {self.best}')
        self.save_audio(model, test_dataloader)
        logger.info(f'Best Loss on Epoch {self.best_epoch}: {self.best}')
        
    def train(self,model, dataloader, criterion, optimizer, epoch,logger):
        _loss = 0
        for x,y in tqdm.tqdm(dataloader):
            x = x.view(x.size(0), -1)
            x = x.to(self.device)
            y = y.to(self.device)
            mask = torch.rand(x.size()) > self.mask_ratio
            x_mask = x * mask
            optimizer.zero_grad()
            output = model(x_mask)
            label = x
            if self.target_mode == 'fft':
                label = fft(x)
                label = torch.Tensor(label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            _loss += loss.item()
        _loss /= len(dataloader)
        logger.info(f'train Loss on Epoch {epoch}: {_loss}')    
        print(f'train Loss on Epoch {epoch}: {_loss}')
        
        
    def test(self, model, dataloader, criterion, logger, epoch):
        _loss = 0
        with torch.no_grad():   
            for x,y in tqdm.tqdm(dataloader):
                x = x.view(x.size(0), -1)
                x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                label = x
                if self.target_mode == 'fft':
                    label = fft(x)
                    label = torch.Tensor(label)
                loss = criterion(output, label)
                _loss += loss.item()
        _loss /= len(dataloader)
        if self.best > _loss:
            self.best = _loss
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
        logger.info('Test Loss on Epoch {:0}: {:.4f}'.format(epoch,_loss))
        print('Test Loss on Epoch {:0}: {:.4f}'.format(epoch,_loss))
    def tsne(self, model, dataloader):
        output_feature = []
        label = []
        for x,y in dataloader:
            x = x.view(x.size(0), -1)
            x = x.to(self.device)
            y = y.to(self.device)
            output = model.extract_feature(x)
            output_feature.append(output)
            label.append(y)
        return torch.cat(output_feature, dim=0), torch.cat(label, dim=0)
    def save_audio(self, model, dataloader):
        for x,y in dataloader:
            x = x.view(x.size(0), -1)
            x = x.to(self.device)
            y = y.to(self.device)
            output = model(x)
            for i in range(10):
                path = f'../ckpt/output/{y[i]}.wav'
                path_x = f'../ckpt/output/{y[i]}_x.wav'
                import soundfile as sf
                sf.write(path_x, x[i].detach().cpu().numpy(), 16000)
                sf.write(path, output[i].detach().cpu().numpy(), 16000)
        print(torch.sum(abs(output[0]-x[0])))
