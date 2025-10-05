# PyTorch YAMNet

This repository implements the YAMNet model for audio classification in PyTorch. YAMNet was originally released in TensorFlow by Google. This implementation is adapted from [Torch AudioSet](https://github.com/w-hc/torch_audioset), which only supports inference using pretrained weights. In contrast, this version adds full support for training from scratch. It also adds support for an enhanced version of YAMNet which replaces the MobileNetV1 backbone with MobileNetV3.

![architecture](https://github.com/user-attachments/assets/03ab628b-8bdc-4574-b92d-0a61683805a2)

## Usage


```bash
pip install -r requirements.txt
./download_esc50.sh
python3 example.py ./ESC-50-master ./log.txt ./ckpt.pt
```


🎧 Sampling Rate Mismatch
Yes, you're correct:

AudioSet uses 48 kHz sampling rate.

ESC-50 and likely your ESC50Artifact dataset use 16 kHz (or 44.1 kHz originally, but often downsampled to 16 kHz for efficiency).

Why this matters:

YAMNet is pretrained on 48 kHz data. Feeding it 16 kHz audio without proper resampling or feature alignment can degrade performance.

If your pipeline skips resampling or uses mismatched mel spectrogram parameters, the model may misinterpret the input.

🧪 Dataset Integrity
You mentioned:

Programmatically injected artifacts with single-label annotations, even though clips may contain multiple artifacts.

This introduces label noise, which can:

Confuse the model during training.

Lead to poor generalization and unstable validation loss.

Suppress accuracy, especially if the model sees conflicting patterns.

Fixes:

Consider multi-label classification if artifacts co-occur.

Audit a subset of the dataset manually to verify label fidelity.

Use label smoothing or soft targets to mitigate noise.

📉 Metric Effectiveness
If you're using accuracy as your primary metric:

It may be misleading for imbalanced or multi-label tasks.

A model predicting only the majority class could still score high accuracy.

Better metrics:

F1-score (macro or weighted): balances precision and recall.

Mean Average Precision (mAP): especially good for multi-label setups.

Confusion matrix: reveals misclassification patterns.

Precision/Recall curves: show threshold sensitivity.

🛠 Model Setup Suggestions
Resample your audio to 48 kHz before feeding it to YAMNet.

Normalize your mel spectrogram parameters to match AudioSet preprocessing.

Try CNN14 or Wavegram-Logmel-CNN14 if YAMNet continues to underperform.

Freeze fewer layers or use gradual unfreezing to retain pretrained knowledge while adapting to your domain.

READ PAPERS & architecture of PANNs and improve dataset quality first