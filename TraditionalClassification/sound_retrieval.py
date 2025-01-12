import numpy as np
import scipy.signal
import librosa
import os
import re
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean, cityblock
from collections import defaultdict
import argparse
from base_spectrum import extract_f0_using_cepstrum


# Step 1: Data Preparation
def load_audio(file_path):
    """Load an audio file and return the signal and sample rate."""
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr


# Step 2: Feature Extraction
def compute_fft(signal, sr):
    """Compute the FFT of an audio signal."""
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    fft_magnitude = np.abs(np.fft.rfft(signal))
    return fft_magnitude


def compute_stft(signal, sr, n_fft=2048, hop_length=512):
    """Compute the STFT of an audio signal."""
    stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    return magnitude


def compute_mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """Compute MFCC features of an audio signal."""
    mfcc_features = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc_features.T


# Step 3: Feature Normalization
def normalize_features(features):
    """Normalize feature vectors to have zero mean and unit variance."""
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)


# Step 4: Database Preparation
def build_database(folder_path, feature_extractor):
    """Extract features for all database audio files."""
    database = {}
    for filename in tqdm(os.listdir(folder_path), desc="Building database"):
        if not filename.startswith("5-"):
            file_path = os.path.join(folder_path, filename)
            signal, sr = load_audio(file_path)
            features = feature_extractor(signal, sr)
            # database[filename] = normalize_features(mfcc_features)
            database[filename] = features
    return database


# Step 5: Query Processing
def process_query(query_file, database, feature_extractor, metric):
    """Process a query and compute similarity scores with the database."""
    query_signal, query_sr = load_audio(query_file)
    query = feature_extractor(query_signal, query_sr)
    # query_features = normalize_features(query_mfcc)
    query_features = query
    # print(query_features)

    scores = {}
    for file_path, db_features in database.items():
        if metric == "cosine":
            score = cosine_similarity(
                query_features.mean(axis=0).reshape(1, -1),
                db_features.mean(axis=0).reshape(1, -1),
            )[0, 0]

            # query_features = query_features.reshape(1, -1)
            # db_features = db_features.reshape(1, -1)
            # score = cosine_similarity(query_features, db_features)[0, 0]
        elif metric == "euclidean":
            # score = -euclidean(query_features.mean(axis=0), db_features.mean(axis=0))

            query_features = query_features.flatten()
            db_features = db_features.flatten()
            score = -euclidean(query_features, db_features)
        elif metric == "manhattan":
            # score = -cityblock(query_features.mean(axis=0), db_features.mean(axis=0))

            query_features = query_features.flatten()
            db_features = db_features.flatten()
            score = -cityblock(query_features, db_features)
        scores[file_path] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def extract_class(filename):
    pattern = re.compile(r"-(\d+).wav$")
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None


# Step 6: Evaluation
def evaluate_retrieval(query_files, database, feature_extractor, metric, top_k):
    """Evaluate retrieval performance."""
    precision_scores = []
    for query_file in tqdm(query_files, desc="Evaluating"):
        ranked_list = process_query(query_file, database, feature_extractor, metric)
        top_k_files = [x[0] for x in ranked_list[:top_k]]
        # relevant = ground_truth[query_file]
        # hits = len(set(top_k_files) & set(relevant))
        # precision = hits / top_k

        precision = 0
        for file in top_k_files:
            if extract_class(query_file) == extract_class(file):
                precision = 1
                break
        precision_scores.append(precision)
    return np.mean(precision_scores)


def retrieval(query_files, database, feature_extractor, metric, top_k, output_path):
    """Evaluate retrieval performance."""
    for query_file in tqdm(query_files, desc="Evaluating"):
        ranked_list = process_query(query_file, database, feature_extractor, metric)
        top_k_files = [x[0] for x in ranked_list[:top_k]]
        new_dict = {query_file: top_k_files}
        with open(output_path, "a") as f:
            f.write(json.dumps(new_dict) + "\n")
    print(f"Results saved to {output_path}")


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="mfcc",
        choices=["fft", "stft", "mfcc", "base"],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
    )
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # Step 1: Load the dataset
    database_folder = "ESC-50-master/audio"
    query_files = [
        os.path.join(database_folder, f)
        for f in os.listdir(database_folder)
        if f.startswith("5-")
    ]

    if args.feature_extractor == "fft":
        feature_extractor = compute_fft
    elif args.feature_extractor == "stft":
        feature_extractor = compute_stft
    elif args.feature_extractor == "mfcc":
        feature_extractor = compute_mfcc
    elif args.feature_extractor == "base":
        feature_extractor = extract_f0_using_cepstrum

    # Step 2: Build the feature database
    database = build_database(database_folder, feature_extractor)
    # print(len(query_files), len(database))

    # Step 3: Evaluate retrieval precision
    output_path = f"result/{args.feature_extractor}_{args.metric}_{args.top_k}.jsonl"
    retrieval(
        query_files, database, feature_extractor, args.metric, args.top_k, output_path
    )

    # score = evaluate_retrieval(query_files, database, feature_extractor, args.metric, args.top_k)
    # print(score)
