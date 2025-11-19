import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


def compute_feature(audio, sample_rate=44100, n_fft=2048, win_length=2048, hop_length=512, p=2):
    window = torch.hann_window(win_length)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                       window=window, return_complex=True)
    magnitude = stft.abs().squeeze(0)

    freq_bins = magnitude.size(0)
    freqs = torch.linspace(0, sample_rate / 2, steps=freq_bins).unsqueeze(1)

    centroid = (freqs * magnitude).sum(dim=0) / (magnitude.sum(dim=0) + 1e-10)
    deviation = (freqs - centroid.unsqueeze(0)).abs() ** p
    bandwidth = (magnitude * deviation).sum(dim=0) / (magnitude.sum(dim=0) + 1e-10)
    bandwidth = bandwidth ** (1 / p)

    return centroid.unsqueeze(0), bandwidth.unsqueeze(0)


class AudioEmotionDataset(Dataset):
    def __init__(self, root_dir, n_mfcc=10, max_len=300,
                 sr=44100, n_fft=2048, hop_length=512, win_length=2048, p=2):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.p = p
        self.data = []
        self.labels = []

        label_names = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')
        ])

        self.label2idx = {label: idx for idx, label in enumerate(label_names)}
        print("Labels Mapping:", self.label2idx)

        for label in label_names:
            label_path = os.path.join(self.root_dir, label)
            for fname in os.listdir(label_path):
                if fname.endswith('.wav'):
                    file_path = os.path.join(label_path, fname)
                    self.data.append(file_path)
                    self.labels.append(self.label2idx[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        audio, sr = torchaudio.load(file_path)
        if sr != self.sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sr)
            audio = resampler(audio)

        mfcc = T.MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc,
                      melkwargs={'n_fft': self.n_fft, 'hop_length': self.hop_length})(audio)

        centroid, bandwidth = compute_feature(audio, self.sr, self.n_fft, self.win_length, self.hop_length, self.p)

        feature = torch.cat([mfcc.squeeze(0), centroid, bandwidth], dim=0)

        feature = (feature - feature.mean(dim=1, keepdim=True)) / (feature.std(dim=1, keepdim=True) + 1e-6)

        if feature.shape[1] < self.max_len:
            pad = self.max_len - feature.shape[1]
            feature = torch.nn.functional.pad(feature, (0, pad))
        else:
            feature = feature[:, :self.max_len]

        feature = feature.transpose(0, 1)

        return feature, torch.tensor(label, dtype=torch.long)
