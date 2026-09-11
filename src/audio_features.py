import librosa 
import numpy as np


def load_audio(
    path,
    sample_rate=22050
):

    audio, sr = librosa.load(
        path,
        sr=sample_rate,
        mono=True
    )

    return audio, sr


def extract_segment_features(
    audio,
    sample_rate=22050,
    segment_seconds=5
):

    segment_length = (
        sample_rate *
        segment_seconds
    )

    features = []

    for start in range(
        0,
        len(audio),
        segment_length
    ):

        segment = audio[
            start:start + segment_length
        ]

        if len(segment) < sample_rate:
            continue

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sample_rate,
            n_mfcc=13
        )

        mfcc_mean = mfcc.mean(axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=segment,
            sr=sample_rate
        )

        chroma_mean = chroma.mean(axis=1)

        feature = np.concatenate([
            mfcc_mean,
            chroma_mean
        ])

        features.append(feature)

    if len(features) == 0:
        return np.empty((0, 25))

    return np.array(features)