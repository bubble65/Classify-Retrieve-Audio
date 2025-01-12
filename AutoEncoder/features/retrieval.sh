#!/bin/bash

# Define the arrays of variables to loop through
feature_types=("cqcc")
distance_types=("cosine")
sampling_rates=(44100 16000)

# Loop through each combination of variables
for feature in "${feature_types[@]}"; do
    for distance in "${distance_types[@]}"; do
        for sr in "${sampling_rates[@]}"; do
            echo "Running with feature_type=$feature, distance_type=$distance, sr=$sr"
            python features/best.py --feature_type "$feature" --distance_type "$distance" --sr "$sr"
        done
    done
done