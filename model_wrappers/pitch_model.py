from typing import List
import numpy as np

from resampy import resample

from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
from tensorflow.keras.models import Model


class PitchModel:
    def __init__(self, batch_size: int = 16,
                 pitch_thresh: float = 0.65):
        """
        :param pitch_thresh: all signals with less confidence are marked as silence
        """
        self.model_sr = 16000
        self.model_inp_size = 1024
        self.model = self._build_and_load_model("full", "../models/model-full.h5")

        self.batch_size = batch_size
        self.pitch_thresh = pitch_thresh

    def detect_pitches(self, vocals: np.array, sample_rate: int) -> List[float]:
        pitches = []
        confidences = []
        step_size = self.model_inp_size * self.batch_size
        mono_vocals_resampled = resample(vocals, sample_rate, self.model_sr).reshape(-1, 1)

        for split_idx in range(0, len(mono_vocals_resampled), step_size):
            batch_pitch, batch_confidence = self.detect_pitch(mono_vocals_resampled[split_idx: split_idx + step_size])
            pitches += list(batch_pitch)
            confidences += list(batch_confidence)

        pitches = np.array(pitches)
        confidences = np.array(confidences)

        pitches[confidences < 0.65] = None

        pitches = pitches.astype(np.float32)
        pitches = [elem for elem in pitches]
        return pitches

    def detect_pitch(self, signal, pitch_model):
        split_idxs = np.arange(self.model_inp_size, len(signal), self.model_inp_size)
        frames = np.split(signal, split_idxs)

        last = frames[-1]

        if len(last) < 1024:
            need_to_pad = 1024 - len(last)
            right_zeros = need_to_pad // 2
            left_zeros = need_to_pad - right_zeros
            frames[-1] = np.concatenate([np.zeros((left_zeros, 1)), last, np.zeros((right_zeros, 1))])

        frames = np.concatenate(frames, axis=1)
        frames = frames.transpose(1, 0)  # had shape (1024, n_samples), converted to (n_samples, 1024)

        # normalize each frame -- this is expected by the model
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        std = np.std(frames, axis=1)[:, np.newaxis]
        std[std == 0] = 1e-10
        frames /= std

        model_preds = pitch_model(frames, training=False)  # , workers=-1, use_multiprocessing=True)
        model_preds = model_preds.numpy()

        # initially has out shape (length, 360), reducing
        too_low_too_high_mask = np.array([True] * 80 + [False] * 140 + [True] * 140)
        model_preds[:, too_low_too_high_mask] = 0

        #     print("time needed", time.time() - start_time)
        batch_pitch = model_preds.argmax(axis=1)
        confidence = model_preds.max(axis=1)

        return batch_pitch, confidence

    @staticmethod
    def _build_and_load_model(model_capacity, filename):
        """
        Build the CNN model and load the weights
        Parameters
        ----------
        model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
            String specifying the model capacity, which determines the model's
            capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
            or 32 (full). 'full' uses the model size specified in the paper,
            and the others use a reduced number of filters in each convolutional
            layer, resulting in a smaller model that is faster to evaluate at the
            cost of slightly reduced pitch estimation accuracy.
        Returns
        -------
        model : tensorflow.keras.models.Model
            The pre-trained keras model loaded in memory
        """

        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(1024,), name='input', dtype='float32')
        y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(f, (w, 1), strides=s, padding='same',
                       activation='relu', name="conv%d" % l)(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                          name="conv%d-maxpool" % l)(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(360, activation='sigmoid', name="classifier")(y)

        model = Model(inputs=x, outputs=y)

        model.load_weights(filename)
        model.compile('adam', 'binary_crossentropy')

        return model

