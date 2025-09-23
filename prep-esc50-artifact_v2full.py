import os
import tqdm
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import pandas as pd

# --------------------------
# Helper
# --------------------------
def save_audio(y, sr, path):
    sf.write(path, y, sr)

# --------------------------
# Artifact functions
# --------------------------

def add_hiss(y, snr_db=15):
    """Add white noise with small random variations."""
    # base noise
    noise = np.random.randn(len(y))
    # normalize noise to RMS=1
    noise /= np.sqrt(np.mean(noise**2))
    signal_power = np.mean(y**2)
    noise_power = signal_power / (10**(snr_db/10))
    # scale noise randomly ±10%
    noise *= np.sqrt(noise_power) * np.random.uniform(0.9,1.1)
    return y + noise

def add_hum(y, sr, freq=60, strength=0.1):
    """Add low-frequency hum with harmonics."""
    t = np.arange(len(y)) / sr
    # base 50/60 Hz
    hum = np.sin(2*np.pi*freq*t)
    # add harmonics (optional for realism)
    hum += 0.5*np.sin(2*np.pi*2*freq*t + np.random.rand()*np.pi)
    hum += 0.25*np.sin(2*np.pi*3*freq*t + np.random.rand()*np.pi)
    hum /= np.max(np.abs(hum))
    # apply random strength variation ±20%
    return y + strength * np.random.uniform(0.8,1.2) * hum

def add_clipping(y, strength=0.7):
    threshold = strength * np.max(np.abs(y)) * np.random.uniform(0.9,1.1)
    return np.clip(y, -threshold, threshold)

def add_distortion(y, strength=1.0):
    # soft clipping with small random variation
    return np.tanh(strength * np.random.uniform(0.9,1.1) * y)

def add_echo(y, sr, delay=0.25, decay=0.5, repeats=2):
    """Echo with multiple repeats and small random variation."""
    delay_samples = int(sr * delay * np.random.uniform(0.9,1.1))
    output = y.copy()
    for i in range(1, repeats+1):
        start = i*delay_samples
        if start < len(y):
            output[start:] += (decay**i) * np.random.uniform(0.9,1.1) * y[:-start]
    return output

def add_lowpass(y, sr, cutoff=2000, order=6):
    cutoff_hz = cutoff * np.random.uniform(0.9,1.1)  # small randomization
    nyq = 0.5 * sr
    norm_cutoff = cutoff_hz / nyq
    b, a = signal.butter(order, norm_cutoff, btype='low')
    return signal.lfilter(b, a, y)

# --------------------------
# ESC50 Artifact Generator
# --------------------------
def generate_esc50_artifacts(esc50_df, input_dir="ESC-50-master/audio", output_root="ESC50Artifact"):
    
    audio_dir = os.path.join(output_root, "audio")
    meta_dir = os.path.join(output_root, "meta")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    artifact_list = ["hiss","hum","clipping","distortion","echo","lowpass"]
    rows = []

    for _, row in tqdm.tqdm(esc50_df.iterrows(), total=len(esc50_df)):
        file = row['filename']
        base_class = row['category']
        fold = row['fold']
        
        y, sr = librosa.load(os.path.join(input_dir, file), sr=16000)
        
        # Save clean
        out_file = f"{file.replace('.wav','')}_clean.wav"
        save_audio(y, sr, os.path.join(audio_dir, out_file))
        rows.append([out_file, fold, base_class, "clean", "none", file])
        
        # Apply artifacts
        for artifact in artifact_list:
            if artifact == "hiss":
                y_aug = add_hiss(y, snr_db=np.random.choice([10,15,20]))
                strength_label = "medium"
            elif artifact == "hum":
                y_aug = add_hum(y, sr, freq=np.random.choice([50,60]), strength=np.random.uniform(0.1,0.2))
                strength_label = "medium"
            elif artifact == "clipping":
                y_aug = add_clipping(y, strength=np.random.uniform(0.6,0.8))
                strength_label = "high"
            elif artifact == "distortion":
                y_aug = add_distortion(y, strength=np.random.uniform(0.9,1.1))
                strength_label = "high"
            elif artifact == "echo":
                y_aug = add_echo(y, sr, delay=np.random.uniform(0.2,0.3), decay=np.random.uniform(0.4,0.6), repeats=2)
                strength_label = "medium"
            elif artifact == "lowpass":
                y_aug = add_lowpass(y, sr, cutoff=np.random.uniform(1800,2200))
                strength_label = "medium"
            
            # Normalize output to [-1,1]
            y_aug /= np.max(np.abs(y_aug) + 1e-9)
            
            out_file = f"{file.replace('.wav','')}_{artifact}.wav"
            save_audio(y_aug, sr, os.path.join(audio_dir, out_file))
            rows.append([out_file, fold, base_class, artifact, strength_label, file])

    # Save metadata
    out_meta = pd.DataFrame(
        rows,
        columns=["filename","fold","base_class","artifact_label","artifact_strength","src_file"]
    )
    out_meta.to_csv(os.path.join(meta_dir, "esc50_artifact.csv"), index=False)
    print(f"ESC50 Artifact dataset generated in {output_root} with metadata.")
    
if __name__ == "__main__":
    esc50_meta = pd.read_csv("ESC-50-master/meta/esc50.csv")  # adjust path if needed
    generate_esc50_artifacts(esc50_meta)