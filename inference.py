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
model.load_state_dict(torch.load("checkpoints/yamnet_1.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 2. Load & preprocess an audio file
# -----------------------------
audio_path = "ESC50Artifact/audio/1-1791-A-26_hum.wav"  # replace with your audio file
waveform, sr = torchaudio.load(audio_path)

# Resample if needed
if sr != params.SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, sr, params.SAMPLE_RATE)

# Ensure mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

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

print(f"Predicted class: {pred_class}, confidence: {confidence:.3f}")

# -----------------------------
# 4. (Optional) Visualize waveform & probabilities
# -----------------------------
plt.figure(figsize=(12,4))
plt.plot(waveform.squeeze().cpu().numpy())
plt.title("Waveform")
plt.show()

plt.figure(figsize=(8,4))
plt.bar(range(num_classes), avg_probs.cpu().numpy())
plt.title("Prediction Probabilities")
plt.xlabel("Class ID")
plt.ylabel("Probability")
plt.show()
