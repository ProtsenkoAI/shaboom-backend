import demucs
import demucs.utils
import demucs.separate
import demucs.pretrained
import numpy as np

from typing import Tuple


class VoiceExtrModel:
    def __init__(self, n_shifts: int = 1):
        """
        :param n_shifts: number of shifts in demucs model. Used for time equivariant.
        """
        self.sources_model = demucs.pretrained.load_pretrained("demucs")
        self.model_sample_rate = self.sources_model.samplerate
        self.device = "cpu"
        self.n_channels = 2

        self.n_shifts = n_shifts

    def get_vocals(self, pth: str) -> Tuple[np.array, int]:
        audio = self._load(pth)
        vocals = self._extract_audio(audio)
        return vocals, self.model_sample_rate

    def _load(self, pth: str) -> np.array:
        return demucs.separate.load_track(pth, self.device, self.n_channels,
                                          self.model_sample_rate)

    def _extract_audio(self, mix: np.array) -> np.array:
        """Returns signal of same shape, with voice track
        extracted.
        """
        ref_for_stats = mix.mean(0)
        normalized_mix = (mix - ref_for_stats.mean()) / ref_for_stats.std()

        all_sources_separated = demucs.utils.apply_model(
            self.sources_model, normalized_mix, shifts=self.n_shifts, split=True,
            overlap=0.25, progress=True
        )
        vocals_source_idx = self.sources_model.sources.index("vocals")
        return all_sources_separated[vocals_source_idx]
