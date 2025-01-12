model=("cnn" "crnn" "attention_cnn" "lstm" "resnet")
wl=('1024' '2048' '4096')
for m in "${model[@]}"; do
    echo "Running experiment with model: $m"
    for w in "${wl[@]}"; do
        python main.py --method raw --batch_size 128 --window_length $w --model $m
        python main.py --method fft --batch_size 200 --window_length $w --model $m
        python main.py --method mel --batch_size 1500 --window_length $w --model $m
        python main.py --method mfcc --batch_size 100 --transform dst --window_length $w --model $m
        python main.py --method mfcc --batch_size 100 --transform dct --window_length $w --model $m
        python main.py --method mfcc_derivatives --batch_size 100 --transform dst --window_length $w --model $m
        python main.py --method mfcc_derivatives --batch_size 100 --transform dct --window_length $w --model $m
        python main.py --method stft --batch_size 100 --window_length $w --model $m
        python main.py --method stft_DCT --batch_size 100 --window_length $w --model $m
        python main.py --method stft_derivatives --batch_size 100 --window_length $w --model $m
    done
done
# python main.py --method raw --batch_size 128 
# python main.py --method fft --batch_size 200
# python main.py --method mel --batch_size 1500
# python main.py --method mfcc --batch_size 100 --transform dst
# python main.py --method mfcc --batch_size 100 --transform dct
# python main.py --method mfcc_derivatives --batch_size 100 --transform dst
# python main.py --method mfcc_derivatives --batch_size 100 --transform dct
# python main.py --method stft --batch_size 100
# python main.py --method stft_DCT --batch_size 100
# python main.py --method stft_derivatives --batch_size 100
