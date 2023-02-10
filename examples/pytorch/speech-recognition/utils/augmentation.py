#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    # FrequencyMask,
    Gain,
    PitchShift,
    PolarityInversion,
    # TimeMask,
    TimeStretch,
    OneOf,
)


class SpeechAugmentation:
    def __init__(self, musan_dir="/home/bhuang/corpus/speech/public/musan_wo_speech"):
        # todo: reverb
        self.transform = Compose(
            [
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
                Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                OneOf(
                    [
                        # bh: slow
                        AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0, noise_transform=PolarityInversion(), p=1.0),
                        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
                    ],
                    p=0.2,
                ),
                # TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.05),
                # FrequencyMask(min_frequency_band=0.2, max_frequency_band=0.4, p=0.05),
            ]
        )

    def __call__(self, waveform, sample_rate):
        waveform = self.transform(waveform, sample_rate=sample_rate)
        return waveform
