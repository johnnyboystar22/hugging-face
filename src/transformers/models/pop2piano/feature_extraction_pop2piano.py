# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Feature extractor class for Pop2Piano"""

from typing import List, Optional, Union

import librosa
import numpy as np
import scipy
import torch
from torch.nn.utils.rnn import pad_sequence

from ...audio_utils import fram_wave, get_mel_filter_banks, stft
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import OptionalDependencyNotAvailable, TensorType, is_essentia_available, logging


try:
    if not is_essentia_available:
        raise OptionalDependencyNotAvailable()
except ImportError:
    raise ImportError("There was an error while importing essentia!")
else:
    import essentia
    import essentia.standard


logger = logging.get_logger(__name__)


class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pop2Piano feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
    This class extracts rhythm and does preprocesses before being passed through the transformer model.
        sampling_rate (`int`, *optional*, defaults to 22050):
            Target Sampling rate of audio signal. It's the sampling rate that we forward to the model.
        padding_value (`int`, *optional*, defaults to 0):
            Padding value used to pad the audio. Should correspond to silences.
        fft_window_size (`int`, *optional*, defaults to 4096):
            Size of the window om which the Fourier transform is applied.
        hop_length (`int`, *optional*, defaults to 1024):
            Step between each window of the waveform.
        frequency_min (`float`, *optional*, defaults to 10.0):
            Minimum frequency.
        nb_mel_filters (`int`, *optional*, defaults to 512):
            Number of Mel filers to generate.
        n_bars (`int`, *optional*, defaults to 2):
            Determines interval between each sequence.
    """
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate: int = 22050,
        padding_value: int = 0,
        fft_window_size: int = 4096,
        hop_length: int = 1024,
        frequency_min: float = 10.0,
        nb_mel_filters: int = 512,
        n_bars: int = 2,
        feature_size=None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.fft_window_size = fft_window_size
        self.hop_length = hop_length
        self.frequency_min = frequency_min
        self.nb_mel_filters = nb_mel_filters
        self.n_bars = n_bars

    def log_mel_spectogram(self, sequence):
        """Generates MelSpectrogram then applies log base e."""

        mel_fb = get_mel_filter_banks(
            nb_frequency_bins=(self.fft_window_size // 2) + 1,
            nb_mel_filters=self.nb_mel_filters,
            frequency_min=self.frequency_min,
            frequency_max=float(self.sampling_rate // 2),
            sample_rate=self.sampling_rate,
            norm=None,
            mel_scale="htk",
        ).astype(np.float32)

        spectrogram = []
        for seq in sequence:
            window = np.hanning(self.fft_window_size + 1)[:-1]
            framed_audio = fram_wave(seq, self.hop_length, self.fft_window_size)
            spec = stft(framed_audio, window, fft_window_size=self.fft_window_size)
            spec = np.abs(spec) ** 2.0
            spectrogram.append(spec)

        spec_shape = spec.shape
        spectrogram = np.array(spectrogram).reshape(-1, *spec_shape)
        log_melspec = np.log(
            np.clip(
                np.transpose(np.matmul(np.transpose(spectrogram, (0, -1, -2)), mel_fb), (0, -1, -2)),
                a_min=1e-6,
                a_max=None,
            )
        )

        return log_melspec

    def extract_rhythm(self, raw_audio):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as
        tempo in bpm for an audio signal. For more information please visit
        https://essentia.upf.edu/reference/std_RhythmExtractor2013.html .
        """
        essentia_tracker = essentia.standard.RhythmExtractor2013(method="multifeature")
        bpm, beat_times, confidence, estimates, essentia_beat_intervals = essentia_tracker(raw_audio)

        return bpm, beat_times, confidence, estimates, essentia_beat_intervals

    def interpolate_beat_times(self, beat_times, steps_per_beat, extend=False):
        beat_times_function = scipy.interpolate.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )
        if extend:
            beat_steps_8th = beat_times_function(np.linspace(0, beat_times.size, beat_times.size * steps_per_beat + 1))
        else:
            beat_steps_8th = beat_times_function(
                np.linspace(0, beat_times.size - 1, beat_times.size * steps_per_beat - 1)
            )
        return beat_steps_8th

    def extrapolate_beat_times(self, beat_times, n_extend=1):
        beat_times_function = scipy.interpolate.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )
        ext_beats = beat_times_function(np.linspace(0, beat_times.size + n_extend - 1, beat_times.size + n_extend))

        return ext_beats

    def preprocess_mel(
        self,
        audio,
        beatstep,
        n_bars,
        padding_value,
    ):
        """Preprocessing for log-mel spectrogram"""

        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        ext_beatstep = self.extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        batch = []
        for i in range(0, n_target_step, n_steps):
            start_idx = i
            end_idx = min(i + n_steps, n_target_step)

            start_sample = int(ext_beatstep[start_idx] * self.sampling_rate)
            end_sample = int(ext_beatstep[end_idx] * self.sampling_rate)
            feature = audio[start_sample:end_sample]
            batch.append(feature)
        batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)

        return batch, ext_beatstep

    def single_preprocess(
        self,
        beatstep,
        audio=None,
    ):
        """Preprocessing method for a single sequence."""

        if audio is not None:
            if len(audio.shape) != 1:
                raise ValueError(
                    f"Expected `audio` to be a single channel audio input of shape to be (n, ) but found {audio.shape}."
                )

        if beatstep[0] > 0.01:
            logger.warning(f"`beatstep[0]` is not 0. All beatstep will be shifted by {beatstep[0]}.")
            beatstep = beatstep - beatstep[0]

        batch, ext_beatstep = self.preprocess_mel(
            audio,
            beatstep,
            n_bars=self.n_bars,
            padding_value=self.padding_value,
        )

        return batch, ext_beatstep

    def __call__(
        self,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray]],
        sampling_rate: int,
        steps_per_beat: int = 2,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
        Main method to featurize and prepare for the model one sequence. Please note that `Pop2PianoFeatureExtractor`
        only accepts one raw_audio at a time.
            raw_audio (`np.ndarray`, `List`):
                Denotes the raw_audio.
            sampling_rate (`int`):
                Denotes the Sampling Rate of `raw_audio`.
            steps_per_beat (`int`, *optional*, defaults to 2):
                Denotes Steps per beat.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to 'pt'):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        is_batched = bool(
            isinstance(raw_audio, (list, tuple))
            and (isinstance(raw_audio[0], np.ndarray) or isinstance(raw_audio[0], (tuple, list)))
        )
        if is_batched:
            raise ValueError(
                "Pop2PianoFeatureExtractor only takes one raw_audio at a time, if you want to extract features from more than a single audio then you might need to call it multiple times."
            )

        # If it's [np.ndarray]
        if isinstance(raw_audio, list) and isinstance(raw_audio[0], np.ndarray):
            raw_audio = raw_audio[0]

        bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(raw_audio=raw_audio)
        beatsteps = self.interpolate_beat_times(beat_times, steps_per_beat, extend=True)

        if self.sampling_rate != sampling_rate and self.sampling_rate is not None:
            # Change sampling_rate to self.sampling_rate
            raw_audio = librosa.core.resample(
                raw_audio, orig_sr=sampling_rate, target_sr=self.sampling_rate, res_type="kaiser_best"
            )

        sampling_rate = self.sampling_rate
        start_sample = int(beatsteps[0] * sampling_rate)
        end_sample = int(beatsteps[-1] * sampling_rate)
        _audio = torch.from_numpy(raw_audio)[start_sample:end_sample]

        batch, ext_beatstep = self.single_preprocess(
            beatstep=beatsteps - beatsteps[0],
            audio=_audio,
        )

        # Apply LogMelSpectogram
        batch = np.transpose(self.log_mel_spectogram(batch), (0, -1, -2))

        output = BatchFeature(
            {
                "input_features": batch,
                "beatsteps": beatsteps,
                "ext_beatstep": ext_beatstep,
            }
        )

        if return_tensors is not None:
            output = output.convert_to_tensors(return_tensors)

        return output
