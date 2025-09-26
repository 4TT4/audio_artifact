import torch
import torchaudio
import matplotlib.pyplot as plt
from srcv2full.model import YAMNet
from srcv2full.feature_extraction import WaveformToMelSpec
import srcv2full.params as params

# -----------------------------
# 1. Load your trained model
# -----------------------------
num_classes = params.NUM_CLASSES  # 7 for ESC50Artifact
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YAMNet()
model.load_state_dict(torch.load("checkpoints/yamnet_finetune_esc50artifact_frozen_-3.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Load & preprocess an audio file
# -----------------------------
audio_path = "ESC50Artifact/audio/5-262641-A-15_hiss.wav"  # replace with your audio file
waveform, sr = torchaudio.load(audio_path)

# Resample if needed
if sr != params.SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, sr, params.SAMPLE_RATE)

# Ensure mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
waveform = waveform.to(device)  # move waveform to the same device as model and transform

# Convert waveform to Mel Spectrogram chunks
waveform_to_mel = WaveformToMelSpec(device=device)
x_chunks, mel_spectrogram = waveform_to_mel(waveform, params.SAMPLE_RATE)

# Move chunks to device
x_chunks = x_chunks.to(device)

# -----------------------------
# 3. Run forward pass
# -----------------------------
with torch.no_grad():
    logits = model(x_chunks)  # [num_chunks, num_classes]
    probs = torch.softmax(logits, dim=-1)

# Average predictions across chunks
avg_probs = probs.mean(dim=0)
pred_class = torch.argmax(avg_probs).item()
confidence = avg_probs[pred_class].item()

# Add mapping of class indices to labels
class_map = {0: 'clean', 1: 'clipping', 2: 'distortion', 3: 'echo', 4: 'hiss', 5: 'hum', 6: 'lowpass'}
pred_class_name = class_map.get(pred_class, "Unknown")

print(f"Predicted class: {pred_class_name} ({pred_class}), confidence: {confidence:.3f}")

# -----------------------------
# 4. (Optional) Visualize waveform, spectrogram & probabilities
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(waveform.squeeze().cpu().numpy())
plt.title(f"Waveform\nPredicted class: {pred_class_name} ({pred_class}), Confidence: {confidence:.3f}")
plt.show()

plt.figure(figsize=(12,4))
plt.imshow(mel_spectrogram.squeeze(), aspect='auto', origin='lower', cmap='Reds')
plt.title("Mel Spectrogram (Reds)")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.colorbar()
plt.show()

plt.figure(figsize=(12,4))
plt.bar(range(num_classes), avg_probs.cpu().numpy())
plt.xticks(range(num_classes), list(class_map.values()))
plt.title("Prediction Probabilities")
plt.xlabel("Class Name")
plt.ylabel("Probability")
plt.show()
