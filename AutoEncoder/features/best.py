import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from numba import njit, prange
from typing import List, Tuple, Dict
import math
import argparse
@njit(parallel=True)
def dtw_distance(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.full((n, m), np.inf)
    dtw_matrix[0, 0] = abs(x[0] - y[0])
    
    for i in range(1, n):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + abs(x[i] - y[0])
    
    for j in range(1, m):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + abs(x[0] - y[j])
    
    for i in range(1, n):
        for j in range(1, m):
            cost = abs(x[i] - y[j])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix[n-1, m-1]

class FeatureExtractor:
    @staticmethod
    def extract_mfcc_features(audio_path: str, sr: int = 44100, duration: int = 5,
                            n_mfcc: int = 13, n_fft: int = 4096, 
                            hop_length: int = 1024, n_mels: int = 64) -> np.ndarray:
        """Extract MFCC features"""
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr,
            n_mfcc=n_mfcc,         
            n_fft=n_fft,           
            hop_length=hop_length,  
            n_mels=n_mels,         
            fmin=0,             
            fmax=sr//2,             
            window='hamming'       
        )
        
        features = mfccs.T.reshape(-1).astype(np.float64)
        return features
    def extract_stft_features(audio_path: str, sr: int = 44100, duration: int = 5,
                            n_mfcc: int = 13, n_fft: int = 4096, 
                            hop_length: int = 1024, n_mels: int = 64) -> np.ndarray:
        """Extract STFT features"""
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        y = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        features = np.abs(y).reshape(-1).astype(np.float64)
        return features

    
    @staticmethod
    def extract_f0_features(audio_path: str, sr: int = 44100, duration: int = 5,
                          n_fft: int = 2048, hop_length: int = 1024) -> np.ndarray:
        """Extract F0 features using cepstrum"""
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        # 预加重
        y_pre = librosa.effects.preemphasis(y)
        # 分帧
        frames = librosa.util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
        # 加窗
        window = np.hamming(n_fft)
        frames = frames * window[:, np.newaxis]
        # 计算频谱
        spectrum = np.fft.rfft(frames, axis=0)
        # 取对数幅度谱
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        # 计算倒谱
        cepstrum = np.fft.irfft(log_spectrum, axis=0)
        # 寻找倒谱中的峰值
        # peaks = np.mean(cepstrum[1:n_fft // 2], axis=0) 
        peaks = np.argmax(cepstrum[1:n_fft // 2], axis=0) + 1
        # 计算基频
        f0 = 1 / peaks
        f0 = peaks
        return f0.astype(np.float64)
    
    @staticmethod
    def extract_f1_features(audio_path: str, sr: int = 44100, duration: int = 5,
                          n_fft: int = 2048, hop_length: int = 1024) -> np.ndarray:
        """Extract F0 features using cepstrum"""
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        # 预加重
        y_pre = librosa.effects.preemphasis(y)
        # 分帧
        frames = librosa.util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
        # 加窗
        window = np.hamming(n_fft)
        frames = frames * window[:, np.newaxis]
        # 计算频谱
        spectrum = np.fft.rfft(frames, axis=0)
        # 取对数幅度谱
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        # 计算倒谱
        cepstrum = np.fft.irfft(log_spectrum, axis=0)
        # 寻找倒谱中的峰值
        # peaks = np.mean(cepstrum[1:n_fft // 2], axis=0) +1
        peaks = np.argmax(cepstrum[1:n_fft // 2], axis=0) + 1
        # 计算基频
        f0 = sr / peaks
        # f0 = peaks
        return f0.astype(np.float64)
    
    @staticmethod
    def extract_CQCC_features(audio_path: str, sr: int = 44100, duration: int = 5,
                          n_fft: int = 2048, hop_length: int = 1024,N =12) -> np.ndarray:
        # y, sr = librosa.load(audio_path, sr=sr, duration=duration)

        import numpy as np 
        import librosa 
        from scipy.fft import fft 
        import matplotlib.pyplot as plt 
        def compute_mel_spectrogram(audio_file, n_mels=23, n_fft=2048, hop_length=512): 
            """ 计算 Mel 频谱图 :param audio_file: 输入的音频文件路径 :param n_mels: Mel频带数目 :param n_fft: FFT窗口大小 :param hop_length: 帧移大小 :return: Mel频谱图 """ 
            y, sr = librosa.load(audio_file, sr=None) 
            # 计算短时傅里叶变换（STFT） 
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length) 
            # # 转换到Mel频域 
            mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(D), sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length) 
            return mel_spectrogram, sr 
        def compute_cqcc(mel_spectrogram, n_mels=23, n_fft=2048): 
            """ 基于Mel频谱图计算CQCC特征 :param mel_spectrogram: 输入的Mel频谱图 :param n_mels: Mel频带数目 :param n_fft: FFT窗口大小 :return: CQCC特征 """
             # 计算梅尔频谱图的对数 
            log_mel_spectrogram = np.log(mel_spectrogram + 1e-6) 
            # 避免对数为负无穷大 
            # # 进行复数傅里叶变换 
            cqcc_features = fft(log_mel_spectrogram, axis=-1) 
            # 取复数结果的实部和虚部 
            real_part = np.real(cqcc_features) 
            imag_part = np.imag(cqcc_features) 
            # 结合实部和虚部得到最终特征 
            cqcc_features = np.concatenate([real_part, imag_part], axis=0) 
            return cqcc_features
        mel_spectrogram, sr = compute_mel_spectrogram(audio_path, n_mels=13, n_fft=n_fft, hop_length=hop_length)
        cqcc_features = compute_cqcc(mel_spectrogram, n_mels=13, n_fft=n_fft)
        features = cqcc_features.T.reshape(-1).astype(np.float64)
        # # 计算常数 Q 变换
        # CQT = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), n_bins=48, bins_per_octave=12)

        # # 计算幅度谱
        # magnitude = np.abs(CQT)

        # # 对幅度谱取对数
        # log_magnitude = np.log(magnitude + 1e-6)

        # # 计算倒谱
        # cepstrum = np.fft.ifft(log_magnitude, axis=0)
        # cepstrum_real = np.real(cepstrum)

        # # 提取前 N 阶倒谱系数（例如 12 阶）

        # cqcc = np.dstack
        # # cqcc = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        # # cqcc = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length) #MRR10 0.2408

        # # 归一化 CQCC 特征
        # # cqcc = (cqcc - np.mean(cqcc)) / np.std(cqcc)
        # features = cqcc.T.reshape(-1).astype(np.float64)
        # # print(features.shape)
        # # from scipy.signal import lfilter
        # # y, sr = librosa.load(audio_path, sr=None)
        # # D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        # # magnitude = np.abs(D)
        # # power_spectrum = magnitude ** 2
        # # mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=24, fmin=0, fmax=sr // 2)
        # # mel_spectrum = np.dot(mel_basis, power_spectrum)
        # # equal_loudness = librosa.core.perceptual_weighting(mel_spectrum, frequencies=librosa.mel_frequencies(n_mels=24), ref=1.0)
        # # compressed_spectrum = np.cbrt(equal_loudness)
        # # order = 12
        # # if not np.isfinite(equal_loudness).all():
        # #     raise ValueError("compressed_spectrum contains non-finite values")
        # # lpc_coefficients = librosa.lpc(compressed_spectrum, order=order)

        # # # 提取 PLP 特征
        # # plp_features = lpc_coefficients[1:]  # 忽略第一个系数（常数项）

        # # # 归一化 PLP 特征
        # # # plp_features = (plp_features - np.mean(plp_features)) / np.std(plp_features)
        # # features = plp_features.T.reshape(-1).astype(np.float64)
        return features
    def extract_spe_features(audio_path: str, sr: int = 44100, duration: int = 5,
                          n_fft: int = 2048, hop_length: int = 1024) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=None)

        # 计算短时傅里叶变换 (STFT)
        D = librosa.stft(y)

        # 计算幅度谱
        magnitude = np.abs(D)

        # 1. 频谱能量 (Spectral Energy)
        spectral_energy = np.sum(magnitude ** 2, axis=0)

        # 2. 频谱质心 (Spectral Centroid)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)

        # 3. 频谱带宽 (Spectral Bandwidth)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude)

        # 4. 频谱对比度 (Spectral Contrast)
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)

        # 5. 频谱平坦度 (Spectral Flatness)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)

        # 6. 频谱滚降点 (Spectral Rolloff)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)
        return spectral_energy.T.astype(np.float64)
        return np.concatenate([
            spectral_energy.T,
            spectral_centroid[0],
            spectral_bandwidth[0],
            spectral_contrast[0],
            spectral_flatness[0],
            spectral_rolloff[0]
        ]).astype(np.float64)

class RetrievalEvaluator:
    def __init__(self, results_path: str):
        self.results = []
        with open(results_path, 'r') as f:
            for line in f:
                self.results.append(json.loads(line.strip()))

    def compute_mrr(self, k: int) -> float:
        # 计算MRR@k
        reciprocal_ranks = []
        for result in self.results:
            query_class = result["class"]
            retrieved = result["query"][:k]
            try:
                rank = retrieved.index(query_class) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
        return np.mean(reciprocal_ranks)
    
    def compute_hit_ratio(self, k: int) -> float:
        # 计算命中率@k
        hit_ratios = []
        hit_ratios_1 = []
        for result in self.results:
            query_class = result["class"]
            retrieved = result["query"][:k]
            hits = sum(1 for label in retrieved if label == query_class)
            hit_ratios.append(hits / k)
            if hits > 0:
                hit_ratios_1.append(1)
            else:
                hit_ratios_1.append(0)  
        return (np.mean(hit_ratios), np.mean(hit_ratios_1))
    
    def evaluate(self, k: int = 10) -> Dict:
        # metrics@k
        return {
            f"MRR@{k}": self.compute_mrr(k),
            f"HitRatio@{k}": self.compute_hit_ratio(k)[0],
            f"HitRatio_easy@{k}": self.compute_hit_ratio(k)[1]
        }

class ESC50Retrieval:
    def __init__(self, data_dir: str, csv_file: str, feature_type: str = 'mfcc',
                 distance_type: str = 'dtw', **feature_params):
        self.data_dir = data_dir
        self.query_data = []
        self.db_data = []
        self.feature_type = feature_type
        self.distance_type = distance_type
        self.feature_params = feature_params
        
        with open(csv_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                filename, fold, target = line.strip().split(",")[:3]
                full_path = os.path.join(data_dir, filename)
                if fold == "5":
                    self.query_data.append((full_path, int(target)))
                else:
                    self.db_data.append((full_path, int(target)))

    def extract_features(self, audio_path: str) -> np.ndarray:
        if self.feature_type == 'mfcc':
            return FeatureExtractor.extract_mfcc_features(audio_path, **self.feature_params)
        elif self.feature_type == 'f0':
            return FeatureExtractor.extract_f0_features(audio_path, **self.feature_params)
        elif self.feature_type == 'cqcc':
            if self.distance_type == 'dtw':
                return FeatureExtractor.extract_CQCC_features(audio_path,N=1, **self.feature_params)
            else:
                return FeatureExtractor.extract_CQCC_features(audio_path,N=12, **self.feature_params)
        elif self.feature_type == 'f1':
            return FeatureExtractor.extract_f1_features(audio_path, **self.feature_params)
        elif self.feature_type == 'stft':
            return FeatureExtractor.extract_stft_features(audio_path, **self.feature_params)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    @staticmethod
    def compute_distances(query_features: np.ndarray, db_features: np.ndarray,
                        use_dtw: bool = True) -> np.ndarray:
        if use_dtw:
            # DTW仍然使用numba并行计算
            n_samples = len(db_features)
            distances = np.zeros(n_samples, dtype=np.float64)
            for i in prange(n_samples):
                distances[i] = dtw_distance(query_features, db_features[i])
            return distances
        else:
            # 余弦相似度使用numpy矩阵运算
            # 计算内积
            dot_product = np.dot(db_features, query_features)
            # 计算模长
            query_norm = np.sqrt(np.sum(query_features ** 2))
            db_norms = np.sqrt(np.sum(db_features ** 2, axis=1))
            # 计算相似度并取负（使得越小越好，与DTW保持一致）
            return -dot_product / (db_norms * query_norm)
    
    def perform_retrieval(self, output_path: str, top_k: int = 20):
        print("Extracting database features...")
        db_features = []
        db_labels = []
        for path, label in tqdm(self.db_data):
            features = self.extract_features(path)
            db_features.append(features)
            db_labels.append(label)
        db_features = np.array(db_features, dtype=np.float64)
        db_labels = np.array(db_labels, dtype=np.int64)
        
        print("Processing queries...")
        results = []
        for query_path, query_label in tqdm(self.query_data):
            query_features = self.extract_features(query_path)
            distances = self.compute_distances(query_features, db_features, 
                                            use_dtw=(self.distance_type == 'dtw'))
            
            top_k_indices = np.argsort(distances)[:top_k]
            top_k_labels = [int(db_labels[i]) for i in top_k_indices]
            
            result = {
                "query": top_k_labels,
                "class": int(query_label)
            }
            results.append(result)
        
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

# [RetrievalEvaluator class remains the same]
def parse_args():
    parser = argparse.ArgumentParser(description='Audio Retrieval System')
    parser.add_argument('--feature_type', type=str, default='f0', choices=['mfcc', 'f0', 'cqcc', 'f1','stft'],
                       help='Feature extraction method')
    parser.add_argument('--distance_type', type=str, default='cosine', choices=['dtw', 'cosine'],
                       help='Distance measure method')
    parser.add_argument('--sr', type=int, default= 16000)
    
    return parser.parse_args()

def main():
    args = parse_args()

    data_dir = "../ESC-50-master/audio"
    csv_file = "../ESC-50-master/meta/esc50.csv"
    if args.sr == 16000:
        hop = 512
        window = 2048
    else:
        hop = 1024
        window = 4096
    results_file = f"results/retrieval_results_1{args.feature_type}_{args.distance_type}_{args.sr}_{window}_{hop}.jsonl"
    
    # Retrieval phase
    retrieval = ESC50Retrieval(
        data_dir, 
        csv_file,
        feature_type=args.feature_type,
        distance_type=args.distance_type,
        sr=args.sr,
        hop_length=hop,
        n_fft=window,
    )
    retrieval.perform_retrieval(results_file, top_k=20)
    
    # Evaluation phase
    evaluator = RetrievalEvaluator(results_file)
    metrics_10 = evaluator.evaluate(k=10)
    metrics_20 = evaluator.evaluate(k=20)
    
    print("\nMetrics@10:")
    for metric, value in metrics_10.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nMetrics@20:")
    for metric, value in metrics_20.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()