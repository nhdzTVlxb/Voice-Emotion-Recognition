import torch
import torchaudio
import torchaudio.transforms as T

from models.model import SERModel
from utils.dataset import compute_feature
from utils.config import Config


def predict(audio_path):
    config = Config()

    model = SERModel(num_classes=len(config.labels)).to(config.device)
    model.load_state_dict(torch.load('./results/best_model.pth', map_location=config.device))
    model.eval()

    audio, sr = torchaudio.load(audio_path)
    if sr != config.sr:
        resampler = T.Resample(orig_freq=sr, new_freq=config.sr)
        audio = resampler(audio)
    audio = audio.float()

    mfcc_transform = T.MFCC(
        sample_rate=config.sr,
        n_mfcc=config.n_mfcc,
        melkwargs={'n_fft': config.n_fft, 'hop_length': config.hop_length}
    )
    mfcc = mfcc_transform(audio)

    centroid, bandwidth = compute_feature(
        audio, sample_rate=config.sr, n_fft=config.n_fft,
        win_length=config.win_length, hop_length=config.hop_length, p=config.p
    )

    feature = torch.cat([mfcc.squeeze(0), centroid, bandwidth], dim=0)
    feature = (feature - feature.mean(dim=1, keepdim=True)) / (feature.std(dim=1, keepdim=True) + 1e-6)

    if feature.shape[1] < config.max_len:
        pad_width = config.max_len - feature.shape[1]
        feature = torch.nn.functional.pad(feature, (0, pad_width), mode='constant', value=0)
    else:
        feature = feature[:, :config.max_len]

    feature = feature.transpose(0, 1).unsqueeze(0).to(config.device)

    with torch.no_grad():
        pred = model(feature)
        pred = torch.argmax(pred, dim=1)

    return config.labels[pred.item()]


if __name__ == "__main__":
    result = predict("/neutral/xxx.wav")
    print(f"预测结果：{result}")
