import numpy as np
import scipy.signal
import librosa
import os
import re
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def compute_fft(signal, sr):
    freqs = np.fft.rfftfreq(len(signal), 1/sr)
    fft_magnitude = np.abs(np.fft.rfft(signal))
    return fft_magnitude

def compute_stft(signal, sr, n_fft=4096, hop_length=1024):
    stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    return magnitude

def compute_mfcc(signal, sr, n_mfcc=13, n_fft=4096, hop_length=1024):
    mfcc_features = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc_features.T

def normalize_features(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

def build_database(folder_path, feature_extractor):
    database = {}
    for filename in tqdm(os.listdir(folder_path), desc="Building database"):
        if not filename.startswith("5-"):
            file_path = os.path.join(folder_path, filename)
            signal, sr = load_audio(file_path)
            features = feature_extractor(signal, sr)
            database[filename] = features
    return database

def process_query(query_file, database, feature_extractor, metric):
    query_signal, query_sr = load_audio(query_file)
    query = feature_extractor(query_signal, query_sr)
    query_features = query

    scores = {}
    for file_path, db_features in database.items():
        if metric == 'cosine':
            query_features = query_features.reshape(1, -1)
            db_features = db_features.reshape(1, -1)
            score = cosine_similarity(query_features, db_features)[0, 0]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores[file_path] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def extract_class(filename):
    pattern = re.compile(r'-(\d+).wav$')
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None

def retrieval(query_files, database, feature_extractor, metric, top_k, output_path):
    for query_file in tqdm(query_files, desc="Evaluating"):
        ranked_list = process_query(query_file, database, feature_extractor, metric)
        top_k_files = [x[0] for x in ranked_list[:top_k]]
        new_dict = {query_file: top_k_files}
        with open(output_path, "a") as f:
            f.write(json.dumps(new_dict) + "\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_extractor", type=str, default="mfcc", choices=["fft", "stft", "mfcc"]) 
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    
    database_folder = "ESC-50-master/audio"
    query_files = [os.path.join(database_folder, f) for f in os.listdir(database_folder) if f.startswith("5-")]

    if args.feature_extractor == "fft":
        feature_extractor = compute_fft
    elif args.feature_extractor == "stft":
        feature_extractor = compute_stft
    elif args.feature_extractor == "mfcc":
        feature_extractor = compute_mfcc

    database = build_database(database_folder, feature_extractor)

    output_path = f"result/{args.feature_extractor}_{args.metric}_{args.top_k}.jsonl"
    retrieval(query_files, database, feature_extractor, args.metric, args.top_k, output_path)
