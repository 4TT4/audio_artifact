import os
import librosa
import soundfile as sf
import numpy as np
from scipy import signal

# --------------------------
# Helper
# --------------------------
def save_audio(y, sr, path):
    sf.write(path, y, sr)

# --------------------------
# Artifact functions
# --------------------------
def add_hiss(y, snr_db=15):
    noise = np.random.randn(len(y))
    noise /= np.sqrt(np.mean(noise**2))
    signal_power = np.mean(y**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise *= np.sqrt(noise_power) * np.random.uniform(0.9,1.1)
    return y + noise

def add_hum(y, sr, freq=60, strength=0.1):
    t = np.arange(len(y)) / sr
    hum = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*2*freq*t + np.random.rand()*np.pi)
    hum /= np.max(np.abs(hum))
    return y + strength * np.random.uniform(0.8,1.2) * hum

def add_clipping(y, strength=0.7):
    threshold = strength * np.max(np.abs(y)) * np.random.uniform(0.9,1.1)
    return np.clip(y, -threshold, threshold)

def add_distortion(y, strength=1.0):
    return np.tanh(strength * np.random.uniform(0.9,1.1) * y)

def add_echo(y, sr, delay=0.25, decay=0.5, repeats=2):
    delay_samples = int(sr * delay * np.random.uniform(0.9,1.1))
    output = y.copy()
    for i in range(1, repeats+1):
        start = i*delay_samples
        if start < len(y):
            output[start:] += (decay**i) * np.random.uniform(0.9,1.1) * y[:-start]
    return output

def add_lowpass(y, sr, cutoff=2000, order=6):
    cutoff_hz = cutoff * np.random.uniform(0.9,1.1)
    nyq = 0.5 * sr
    norm_cutoff = cutoff_hz / nyq
    b, a = signal.butter(order, norm_cutoff, btype='low')
    return signal.lfilter(b, a, y)

# --------------------------
# Test on 3 files
# --------------------------
input_dir = "ESC-50-master/audio"
output_dir = "ESC50ArtifactTest"
os.makedirs(output_dir, exist_ok=True)

test_files = ["1-23222-A-19.wav", "1-85909-A-29.wav", "1-9887-A-49.wav"]  # example 3 files
artifact_list = ["hiss","hum","clipping","distortion","echo","lowpass"]

for file in test_files:
    y, sr = librosa.load(os.path.join(input_dir, file), sr=16000)
    
    # Save clean
    save_audio(y, sr, os.path.join(output_dir, f"{file.replace('.wav','')}_clean.wav"))
    
    # Apply artifacts
    for artifact in artifact_list:
        if artifact == "hiss":
            y_aug = add_hiss(y, snr_db=15)
        elif artifact == "hum":
            y_aug = add_hum(y, sr, freq=60, strength=0.15)
        elif artifact == "clipping":
            y_aug = add_clipping(y, strength=0.7)
        elif artifact == "distortion":
            y_aug = add_distortion(y, strength=1.0)
        elif artifact == "echo":
            y_aug = add_echo(y, sr, delay=0.25, decay=0.5, repeats=2)
        elif artifact == "lowpass":
            y_aug = add_lowpass(y, sr, cutoff=2000)
        
        # Normalize
        y_aug /= np.max(np.abs(y_aug)+1e-9)
        
        out_file = f"{file.replace('.wav','')}_{artifact}.wav"
        save_audio(y_aug, sr, os.path.join(output_dir, out_file))

print(f"Test artifacts generated for {len(test_files)} files in {output_dir}")