import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


class SimpleResnet(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleResnet, self).__init__()
        # Load pretrained ResNet
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, num_classes=50):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, time_steps, mfcc_features)
        if x.dim() == 4:
            x = x.squeeze(1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

    def extract_features(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        lstm_out, _ = self.lstm(x)
        return torch.flatten(lstm_out[:, -1, :], 1)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # 第一个卷积块：处理时频特征
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第二个卷积块：学习更高层特征
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第三个卷积块：进一步提取特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 分类头部
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应池化到固定大小
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # print(f"After flatten: {x.shape}")
        return x


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(AttentionCNN, self).__init__()

        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(128)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # CNN特征提取
        features = self.features(x)  # [B, 128, H, W]

        # 重塑特征图以适应Attention层
        B, C, H, W = features.shape
        features = features.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 应用多头注意力
        attended_features, _ = self.attention(features, features, features)
        attended_features = self.norm(attended_features + features)  # 残差连接

        # 重塑回原来的形状
        attended_features = attended_features.permute(0, 2, 1).reshape(B, C, H, W)

        # 分类
        output = self.classifier(attended_features)
        return output

    def extract_features(self, x):
        features = self.features(x)
        B, C, H, W = features.shape
        features = features.reshape(B, C, -1).permute(0, 2, 1)
        attended_features, _ = self.attention(features, features, features)
        attended_features = self.norm(attended_features + features)
        attended_features = attended_features.permute(0, 2, 1).reshape(B, C, H, W)
        return torch.flatten(attended_features, 1)


class CRNN(nn.Module):
    def __init__(self, num_classes=50):
        super(CRNN, self).__init__()

        # CNN部分用于提取空间特征
        self.cnn = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.rnn = nn.LSTM(
            input_size=128,  # CNN输出的特征维度
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 256*2 (双向LSTM)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # CNN特征提取
        batch_size = x.size(0)
        x = self.cnn(x)

        x = x.permute(0, 2, 3, 1)  # [batch, time, freq, channels]
        x = x.reshape(batch_size, -1, 128)  # [batch, time, features]

        # RNN处理
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 分类
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, -1, 128)
        x, _ = self.rnn(x)
        # print(f"After flatten: {torch.flatten(x, 1).shape}")
        return torch.flatten(x, 1)


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 50,  # ESC-50数据集有50个类别
        n_heads: int = 8,  # 注意力头数
        n_layers: int = 4,  # Transformer编码器层数
        dim_feedforward: int = 512,  # 前馈网络维度
        dropout: float = 0.1,  # dropout比率
    ):
        super().__init__()

        # 处理4D输入的卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 将卷积特征映射到Transformer的维度
        self.input_projection = nn.Linear(32, dim_feedforward)

        # 位置编码（使用可学习的位置编码）
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, dim_feedforward))

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=n_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入tensor, shape: (batch_size, 1, time, freq)
        返回:
            output: 分类结果, shape: (batch_size, n_classes)
        """
        # 获取batch大小
        batch_size = x.size(0)

        # 通过卷积层
        x = self.conv_layer(x)  # (batch_size, 32, freq//4, time//4)

        # 重排张量维度为序列形式
        # 转换为 (batch_size, time//4, freq//4, 32)
        x = x.permute(0, 3, 2, 1)

        # 将最后两个维度展平
        freq_reduced = x.size(2)
        x = x.reshape(batch_size, -1, 32 * freq_reduced)

        # 投影到Transformer维度
        x = self.input_projection(x)

        # 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer编码器
        x = self.transformer_encoder(x)

        # 全局池化（取平均）
        x = torch.mean(x, dim=1)

        # 分类
        output = self.classifier(x)

        return output


class AudioTransformer(nn.Module):
    def __init__(self, num_classes=50, dim=256, num_heads=8, num_layers=6):
        super(AudioTransformer, self).__init__()

        # 初始特征映射
        self.conv_embed = nn.Sequential(
            # 第一层卷积：减少时间和频率维度
            # 输入: [B, 1, 427, 40] -> 输出: [B, 64, 213, 20]
            nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 第二层卷积：进一步减少维度
            # 输入: [B, 64, 213, 20] -> 输出: [B, 64, 106, 10]
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 第三层卷积：最终降维
            # 输入: [B, 64, 106, 10] -> 输出: [B, 128, 53, 5]
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 计算展平后的序列长度
        self.seq_length = 53 * 5  # 最后一层卷积输出的时间步 * 频率维度

        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, dim))

        # 特征投影
        self.linear_projection = nn.Linear(128, dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(0.1), nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # 输入: [B, 1, 427, 40]

        x = self.conv_embed(x)  # -> [B, 128, 53, 5]
        B, C, T, F = x.shape

        # 2. 重塑张量并转置
        x = x.permute(0, 2, 3, 1)  # -> [B, 53, 5, 128]
        x = x.reshape(B, T * F, C)  # -> [B, 265, 128]

        # 3. 线性投影到Transformer维度
        x = self.linear_projection(x)  # -> [B, 265, dim]

        # 4. 添加位置编码
        x = x + self.pos_embedding

        # 5. Transformer处理
        x = self.transformer(x)  # -> [B, 265, dim]

        # 6. 全局池化
        x = x.mean(dim=1)  # -> [B, dim]

        # 7. 分类
        x = self.classifier(x)  # -> [B, num_classes]

        return x
