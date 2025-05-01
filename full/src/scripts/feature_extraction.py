import matplotlib
matplotlib.use('Agg')

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

def extract_features(file_path):
    x, sr = librosa.load(file_path)

    n_frames_desired = 10
    frame_length = x.shape[0] // n_frames_desired
    hop_length = frame_length

    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)

    zcrs = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        zcr = librosa.feature.zero_crossing_rate(
            frame.reshape(1, -1),
            frame_length=frame.size,
            hop_length=frame.size,
            center=False
        )
        zcrs.append(zcr[0, 0])

    zcrs = np.array(zcrs)

    rmss = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        rms = librosa.feature.rms(
            y= frame.reshape(1, -1),
            frame_length=frame.size,
            hop_length=frame.size,
            center=False
        )
        rmss.append(rms[0, 0])

    rmss = np.array(rmss)
    
    energy = np.array([
        sum(abs(x[i:i+frame_length]**2))
        for i in range(0, len(x), hop_length)
    ])

    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

    temporal_centroid = np.sum(times * energy) / np.sum(energy)

    spectral_centroids = librosa.feature.spectral_centroid(y = x, sr = sr)[0]
    spectral_centroids = spectral_centroids.flatten()
    spectral_centroids.shape

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)[0]
    spectral_bandwidth = spectral_bandwidth.flatten()
    spectral_bandwidth.shape


    centroid_mean = np.mean(spectral_centroids)
    centroid_std = np.std(spectral_centroids)
    centroid_min = np.min(spectral_centroids)
    centroid_max = np.max(spectral_centroids)
    centroid_skew = skew(spectral_centroids)
    centroid_kurt = kurtosis(spectral_centroids)

    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)
    bandwidth_min = np.min(spectral_bandwidth)
    bandwidth_max = np.max(spectral_bandwidth)
    bandwidth_skew = skew(spectral_bandwidth)
    bandwidth_kurt = kurtosis(spectral_bandwidth)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=x)

    spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)[0]
    spectral_rolloff_std = np.std(spectral_rolloff, axis=1)[0]

    spectral_flatness_mean = np.mean(spectral_flatness, axis=1)[0]
    spectral_flatness_std = np.std(spectral_flatness, axis=1)[0]

    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)

    features = {}

    for i in range(len(zcrs)):
        features[f'zcrs_{i}'] = zcrs[i][0]

    for i in range(len(rmss)):
        features[f'rmss_{i}'] = rmss[i][0]

    features["centroid_mean"] = centroid_mean
    features["centroid_std"] = centroid_std
    features["centroid_min"] = centroid_min
    features["centroid_max"] = centroid_max
    features["centroid_skew"] = centroid_skew
    features["centroid_kurt"] = centroid_kurt

    features["bandwidth_mean"] = bandwidth_mean
    features["bandwidth_std"] = bandwidth_std
    features["bandwidth_min"] = bandwidth_min
    features["bandwidth_max"] = bandwidth_max
    features["bandwidth_skew"] = bandwidth_skew
    features["bandwidth_kurt"] = bandwidth_kurt

    features["spectral_rolloff_mean"] = spectral_rolloff_mean
    features["spectral_rolloff_std"] = spectral_rolloff_std

    features["spectral_flatness_mean"] = spectral_flatness_mean
    features["spectral_flatness_std"] = spectral_flatness_std

    for i in range(len(mfcc_mean)):
        features[f'mfcc_mean_{i}'] = mfcc_mean[i]
        features[f'mfcc_std_{i}'] = mfcc_std[i]
        features[f'mfcc_min_{i}'] = mfcc_min[i]
        features[f'mfcc_max_{i}'] = mfcc_max[i]


    return features

def extract_features_as_df(file_path):
    # dict to df
    return pd.DataFrame([extract_features(file_path)])

def extract_features_for_split(split_path):
    rows = []
    for emotion in EMOTIONS:
        print(f'Extracting features for {emotion}')
        for file in tqdm(os.listdir(os.path.join(split_path, emotion))):
            label = emotion
            row = {
                'file_name': file,
                **extract_features(os.path.join(split_path, emotion, file)),
                'label': label
            }
            rows.append(row)
    return pd.DataFrame(rows)

def save_spectrogram(split, emotion, audio_path, output_path = None):
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig_spec, axes_spec = plt.subplots(1, 1, figsize=(15, 8))

    img = librosa.display.specshow(D, sr=sr, ax=axes_spec)
    axes_spec.set_xticks([])
    axes_spec.set_yticks([])

    if output_path is None:
        output_path = os.path.join('data', 'spectrograms', split, emotion, f'{os.path.splitext(os.path.basename(audio_path))[0]}.png')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_spectrograms_for_split(split_path):
    for emotion in EMOTIONS:
        for file in tqdm(os.listdir(os.path.join(split_path, emotion))):
            save_spectrogram(split, emotion, os.path.join(split_path, emotion, file))

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        print(f'Extracting features for {split}')
        split_path = os.path.join('data', split)
        features = extract_features_for_split(split_path)
        
        output_path = os.path.join('data', 'features', f'{split}.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features.to_csv(output_path, index=False)

        print(f'Saving spectrograms for {split}')
        save_spectrograms_for_split(split_path)