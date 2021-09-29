import torch
import librosa
import numpy as np
import torchaudio
from torch.nn import functional as F


class VADModel:
    def __init__(self, thresh: float = 0.04):
        """

        :param thresh: minimum model output to mark segment as voiced
        """
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=True)
        self.vad_model_sr = 16000
        self.thresh = thresh

    def mask_out_inactive_vocals(self, vocals: np.array, sample_rate: int) -> np.array:
        mono_vocals = vocals.mean(axis=0, keepdims=False)
        activity_mask = self.get_voice_activity_mask(mono_vocals, sample_rate, thresh=self.thresh)
        mono_vocals = mono_vocals.numpy()

        filling_freq = 5  # it doesn't matter what frequency to pass
        pure_tone = librosa.tone(frequency=filling_freq, sr=sample_rate, length=len(mono_vocals))

        mono_vocals[~activity_mask] = pure_tone[~activity_mask]
        return mono_vocals

    def get_voice_activity_mask(self, wav, sr, thresh=0.02):
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=self.vad_model_sr)
        src_len = len(wav)
        wav = transform(wav)
        sr = self.vad_model_sr

        probs = self.get_voicing_probs(self.vad_model, wav)
        wav_prob = np.full(src_len, 0, dtype=np.float32)

        step = src_len / len(probs)
        start = 0

        for prob in probs.flatten().numpy():
            wav_prob[round(start): round(start + step)] = prob
            start += step
        winsize = sr
        rolling_mean_wav_prob = np.convolve(wav_prob, np.ones(winsize), 'same') / winsize
        return rolling_mean_wav_prob >= thresh

    def get_voicing_probs(self, model, wav, num_samples_per_window: int = 4000, num_steps: int = 8, batch_size=200):
        num_samples = num_samples_per_window
        assert num_samples % num_steps == 0
        step = int(num_samples / num_steps)  # stride / hop

        outs = []
        to_concat = []
        for i in range(0, len(wav), step):
            chunk = wav[i: i + num_samples]
            if len(chunk) < num_samples:
                chunk = F.pad(chunk, (0, num_samples - len(chunk)))
            to_concat.append(chunk.unsqueeze(0))
            if len(to_concat) >= batch_size:
                chunks = torch.Tensor(torch.cat(to_concat, dim=0))
                with torch.no_grad():
                    out = model(chunks)
                outs.append(out)
                to_concat = []

        outs = torch.cat(outs, dim=0)
        return outs[:, 1]  # 1 dim is 'neg' and 'pos' classes, so take pos probability
