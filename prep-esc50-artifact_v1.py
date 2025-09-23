import os
import tqdm
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import augly.audio as audaugs
import pandas as pd

# --------------------------
# Save audio helper
# --------------------------
def save_audio(y, sr, path):
    sf.write(path, y, sr)

# --------------------------
# ARTIFACT FUNCTIONS
# --------------------------

# HISS (white noise)
def add_hiss(y, snr_db=15):
    noise = np.random.randn(len(y))
    noise /= np.sqrt(np.mean(noise**2))
    signal_power = np.mean(y**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise *= np.sqrt(noise_power)
    return y + noise

# HUM (50 or 60 Hz sine wave)
def add_hum(y, sr, freq=60, strength=0.1):
    t = np.arange(len(y)) / sr
    hum = np.sin(2 * np.pi * freq * t)
    hum /= np.max(np.abs(hum))
    return y + strength * hum

# CLIPPING (hard)
def add_clipping(y, strength=0.7):
    threshold = strength * np.max(np.abs(y))
    return np.clip(y, -threshold, threshold)

# DISTORTION (soft)
def add_distortion(y, strength=1.0):
    return np.tanh(strength * y)

# ECHO
def add_echo(y, sr, delay=0.25, decay=0.5):
    delay_samples = int(sr * delay)
    echo_signal = np.zeros(len(y) + delay_samples)
    echo_signal[:len(y)] = y
    echo_signal[delay_samples:] += decay * y
    return echo_signal[:len(y)]

# LOW-PASS FILTER
def add_lowpass(y, sr, cutoff=2000, order=6):
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype='low')
    return signal.lfilter(b, a, y)

# --------------------------
# ESC50 ARTIFACT GENERATOR
# --------------------------
def generate_esc50_artifacts(esc50_df, input_dir="ESC-50-master/audio", output_dir="ESC50Artifact/audio"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    artifact_list = ["hiss","hum","clipping","distortion","echo","lowpass"]
    rows = []

    for _, row in tqdm.tqdm(esc50_df.iterrows(), total=len(esc50_df)):
        file = row['filename']
        base_class = row['category']
        fold = row['fold']
        
        y, sr = librosa.load(os.path.join(input_dir, file), sr=16000)
        
        # Save clean
        out_file = f"{file.replace('.wav','')}_clean.wav"
        save_audio(y, sr, os.path.join(output_dir, out_file))
        rows.append([out_file, fold, base_class, "clean", "none", file])
        
        # Apply artifacts
        for artifact in artifact_list:
            if artifact == "hiss":
                y_aug = add_hiss(y, snr_db=15)
                strength_label = "medium"
            elif artifact == "hum":
                y_aug = add_hum(y, sr, freq=60, strength=0.15)
                strength_label = "medium"
            elif artifact == "clipping":
                y_aug = add_clipping(y, strength=0.7)
                strength_label = "high"
            elif artifact == "distortion":
                y_aug = add_distortion(y, strength=1.0)
                strength_label = "high"
            elif artifact == "echo":
                y_aug = add_echo(y, sr, delay=0.25, decay=0.5)
                strength_label = "medium"
            elif artifact == "lowpass":
                y_aug = add_lowpass(y, sr, cutoff=2000)
                strength_label = "medium"
            
            out_file = f"{file.replace('.wav','')}_{artifact}.wav"
            save_audio(y_aug, sr, os.path.join(output_dir, out_file))
            rows.append([out_file, fold, base_class, artifact, strength_label, file])

    # Save metadata
    out_meta = pd.DataFrame(
        rows,
        columns=["filename","fold","base_class","artifact_label","artifact_strength","src_file"]
    )
    out_meta.to_csv(os.path.join(output_dir, "meta", "esc50_artifact.csv"), index=False)

    print(f"ESC50 Artifact dataset generated in {output_dir} with metadata.")