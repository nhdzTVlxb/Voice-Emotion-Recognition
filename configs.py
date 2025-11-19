import os
import torch

class Config:
    data_path = '/train'
    result_path = '/results'
    log_dir = '/logs'


    labels = ["anger", "fear", "happy", "neutral", "sad"]

    sr = 16000
    n_fft = 400
    hop_length = 160
    n_mels = 64
    n_mfcc = 13
    max_len = 300  # 特征序列最大长度/400

    batch_size = 32
    lr = 1e-4  #1e-5
    epochs = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
