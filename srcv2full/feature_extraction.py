import numpy as np
import torch
from torchaudio.transforms import Resample, MelSpectrogram

import srcv2full.params as params

class WaveformToMelSpec(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        
        window_length_in_samples = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_LENGTH))
        hop_length_in_samples = int(round(params.SAMPLE_RATE * params.STFT_HOP_LENGTH))
        fft_length = 2 ** int(np.ceil(np.log(window_length_in_samples) / np.log(2.0)))
        
        self.transform_to_mel = LogMelSpecTrans(
            params.SAMPLE_RATE,
            n_fft=fft_length,
            win_length=window_length_in_samples,
            hop_length=hop_length_in_samples,
            f_min=params.MEL_MIN_HZ,
            f_max=params.MEL_MAX_HZ,
            n_mels=params.NUM_MEL_BANDS
        )
        
        self.device = device
        if self.device is not None:
            self.to(self.device)
    
    def __call__(self, waveform, sample_rate):
        # Handle multi-channel audio by taking mean
        if waveform.dim() > 1:
            x = waveform.mean(dim=0, keepdim=True)
        else:
            x = waveform.unsqueeze(0)
        
        resampler = Resample(sample_rate, params.SAMPLE_RATE)
        if self.device is not None:
            resampler = resampler.to(self.device)
        
        # Resample and generate mel spectrogram
        x = resampler(x)
        x = self.transform_to_mel(x)
        x = x.squeeze(dim=0).T # (1, C, T) --> (T, C)
        mel_spectrogram = x.cpu().numpy().copy() # for saving spectrogram as images
        
        # Split into chunks
        window_size = int(round(params.PATCH_WINDOW_LENGTH / params.STFT_HOP_LENGTH))
        if params.PATCH_HOP_LENGTH == params.PATCH_WINDOW_LENGTH: # no overlap
            num_chunks = x.shape[0] // window_size
            if num_chunks == 0:  # Handle case where audio is too short
                # Pad the audio to minimum size
                pad_size = window_size - x.shape[0]
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
                num_chunks = 1
            
            num_frames = num_chunks * window_size
            x = x[:num_frames]
            x = x.reshape(num_chunks, 1, window_size, x.shape[-1])
        else:
            patch_hops = int(round(params.PATCH_HOP_LENGTH / params.STFT_HOP_LENGTH))
            num_chunks = max((x.shape[0] - window_size) // patch_hops + 1, 1)
            num_frames = window_size + (num_chunks - 1) * patch_hops
            
            # Pad if necessary
            if x.shape[0] < num_frames:
                pad_size = num_frames - x.shape[0]
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
            
            x = x[:num_frames]
            x_in_frames = x.reshape(-1, x.shape[-1])
            
            # FIXED: Use torch operations instead of numpy
            x_out = torch.empty((num_chunks, window_size, x.shape[-1]), device=self.device, dtype=x.dtype)
            for i in range(num_chunks):
                start_frame = i * patch_hops
                end_frame = start_frame + window_size
                if end_frame <= x_in_frames.shape[0]:
                    x_out[i] = x_in_frames[start_frame:end_frame]
                else:
                    # Handle edge case with padding
                    available = x_in_frames[start_frame:]
                    pad_needed = window_size - available.shape[0]
                    padded = torch.nn.functional.pad(available, (0, 0, 0, pad_needed))
                    x_out[i] = padded
            
            x = x_out.reshape(num_chunks, 1, window_size, x.shape[-1])
            
        return x, mel_spectrogram
        
        
class LogMelSpecTrans(MelSpectrogram):
    def forward(self, waveform):
        spectrogram = self.spectrogram(waveform)
        spectrogram = spectrogram ** 0.5
        
        mel_spectrogram = self.mel_scale(spectrogram)
        mel_spectrogram = torch.log(mel_spectrogram + params.LOG_DELTA)
        
        return mel_spectrogram