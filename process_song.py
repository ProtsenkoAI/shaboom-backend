import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import List

import numpy as np
import json
import math
import os

from model_wrappers import VoiceExtrModel, VADModel, PitchModel


class PitchDetector:
    def process_song(self, inp_pth: str) -> List[float]:
        """Runs whole preprocessing and detection pipeline
        """
        # functions are segmented by models used to prevent RAM-overusing
        vocals, audio_sr = self._get_vocals(inp_pth)
        vocals_cleaned = self._clean_vocals(vocals, audio_sr)
        pitches = self._detect_pitch(vocals_cleaned, audio_sr)
        return pitches

    def _get_vocals(self, pth: str):
        voice_extraction_model = VoiceExtrModel()
        return voice_extraction_model.get_vocals(pth)

    def _clean_vocals(self, vocals: np.array, sample_rate: int) -> np.array:
        voice_activity_model = VADModel()
        return voice_activity_model.mask_out_inactive_vocals(vocals, sample_rate)

    def _detect_pitch(self, vocals: np.array, sample_rate: int) -> List[float]:
        pitch_model = PitchModel()
        return pitch_model.detect_pitches(vocals, sample_rate)


def main():
    """This is a script to get pitches from audio file.
    Script produces .json file in out_path directory containing list of floats (pitches)
    """
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("input_path", help="path to .wav or .mp3 file to process")
    parser.add_argument("out_dir", help="directory where .json file will be placed")

    args = parser.parse_args()

    pitch_detector = PitchDetector()
    pitches = pitch_detector.process_song(args.input_path)

    song_name = args.input_path.split('/')[-1].split('.')[0]
    pitches = [pitch if not math.isnan(pitch) else -1 for pitch in pitches]

    with open(os.path.join(args.out_dir, song_name + "_pitches.json"), "w") as f:
        json.dump(pitches, f)


if __name__ == "__main__":
    main()
