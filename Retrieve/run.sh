mkdir -p logs

feature_extractors=("fft" "stft" "mfcc")
metrics=("cosine")
top_ks=(10 20)

for feature_extractor in "${feature_extractors[@]}"; do
    for metric in "${metrics[@]}"; do
        for top_k in "${top_ks[@]}"; do
            log_file="logs/${feature_extractor}_${metric}_${top_k}.log"
            nohup python sound_retrieval.py --feature_extractor "$feature_extractor" --metric "$metric" --top_k "$top_k" > "$log_file" 2>&1 &
        done
    done
done
