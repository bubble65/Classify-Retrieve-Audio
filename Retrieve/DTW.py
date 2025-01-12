import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from numba import njit, prange
from typing import List, Tuple, Dict
import math

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

class ESC50Retrieval:
    def __init__(self, data_dir: str, csv_file: str):
        self.data_dir = data_dir
        self.query_data = []  # fold 5
        self.db_data = []     # fold 1-4
        
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
        """Extract features from audio file with temporal information preserved"""
        sr = 16000
        duration = 5
        n_mfcc = 13
        n_fft = 2048
        hop_length = 1024
        n_mels = 128
        fmin = 0            
        fmax = 8000
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr,
            n_mfcc=n_mfcc,         
            n_fft=n_fft,           
            hop_length=hop_length,  
            n_mels=n_mels,         
            fmin=fmin,             
            fmax=fmax,             
            window='hamming'       
        )
        features = mfccs.reshape(-1).astype(np.float64)
        return features
    
    @staticmethod
    @njit(parallel=True)
    def compute_dtw_distances(query_features: np.ndarray, db_features: np.ndarray) -> np.ndarray:
        n_samples = len(db_features)
        distances = np.zeros(n_samples, dtype=np.float64)
        
        for i in prange(n_samples):
            distances[i] = dtw_distance(query_features, db_features[i])
        
        return distances
    
    def perform_retrieval(self, output_path: str, top_k: int = 20):
        """Perform retrieval and save results to JSONL file"""
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
            distances = self.compute_dtw_distances(query_features, db_features)
            
            # Get top-k results
            top_k_indices = np.argsort(distances)[:top_k]
            top_k_labels = [int(db_labels[i]) for i in top_k_indices]
            
            # Save results
            result = {
                "query": top_k_labels,
                "class": int(query_label)
            }
            results.append(result)
        
        # Save to JSONL
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

def main():

    data_dir = "./ESC-50-master/audio"
    csv_file = "./ESC-50-master/meta/esc50.csv"
    results_file = "16000_2048_1024.jsonl"
    
    retrieval = ESC50Retrieval(data_dir, csv_file)
    retrieval.perform_retrieval(results_file, top_k=20)
    
if __name__ == "__main__":
    main()