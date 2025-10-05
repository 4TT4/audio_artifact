import os
import streamlit as st
import torch
import librosa
import numpy as np
import plotly.graph_objects as go
import sounddevice as sd
import queue
import time
import torchaudio.functional as TAF
import io
import soundfile as sf
import pandas as pd

# Assume srcv2full is configured in the environment path or files are locally present
from srcv2full.model import YAMNet
from srcv2full.feature_extraction import WaveformToMelSpec
import srcv2full.params as params

# --- Page config ---
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .main > div {display: flex; justify-content: center; align-items: center; height: 100vh;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Constants ---
class_to_color = {
    0: "lightgray", 1: "#FF4136", 2: "#0074D9", 3: "#2ECC40",
    4: "#FF851B", 5: "#B10DC9", 6: "#FFDC00"
}
class_label_map = {
    0: "clean", 1: "clipping", 2: "distortion",
    3: "echo", 4: "hiss", 5: "hum", 6: "lowpass"
}

# --- Cached model loader ---
@st.cache_resource
def load_model_cached(weights_path):
    # Determine device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = YAMNet()
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    model.to(device)
    model.eval()
    return model

# --- Audio processing (Only used for model inference) ---
def process_audio_chunked(model, y, sr, update_progress=None):
    device = next(model.parameters()).device
    waveform = torch.tensor(y)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Resampling occurs here for model inference only
    if sr != params.SAMPLE_RATE:
        waveform = TAF.resample(waveform, orig_freq=sr, new_freq=params.SAMPLE_RATE)
    
    waveform = waveform.to(device)
    waveform_to_mel = WaveformToMelSpec(device=device)
    # The chunking mechanism relies on the resampled audio
    x_chunks, _ = waveform_to_mel(waveform, params.SAMPLE_RATE)
    
    preds = []
    num_chunks = x_chunks.shape[0]
    # Chunk duration is calculated based on the resampled/processed waveform length
    chunk_duration = waveform.shape[1] / params.SAMPLE_RATE / max(num_chunks,1)
    
    for i, chunk in enumerate(x_chunks):
        with torch.no_grad():
            logits = model(chunk.unsqueeze(0))
            pred = torch.softmax(logits, dim=1).argmax().item()
            preds.append(pred)
        if update_progress:
            update_progress((i+1)/num_chunks)
            
    return preds, chunk_duration

# --- Improved timeline ---
def plot_timeline(preds, chunk_duration):
    num_chunks = len(preds)
    x = [chunk_duration]*num_chunks        # width of each bar
    base = [i*chunk_duration for i in range(num_chunks)]  # start positions
    colors = [class_to_color.get(p, "lightgray") for p in preds]
    hover_text = [f"{class_label_map.get(p,'Unknown')}<br>{i*chunk_duration:.2f}s - {(i+1)*chunk_duration:.2f}s" 
                  for i, p in enumerate(preds)]

    fig = go.Figure(go.Bar(
        x=x,
        y=[0.5]*num_chunks,   # vertical position
        base=base,
        width=0.5,
        marker_color=colors,
        opacity=0.7,
        hovertext=hover_text,
        hoverinfo="text",
        orientation='h',
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor='lightgray'),
        yaxis=dict(visible=False, range=[0,1]),  # keep bars centered
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        title="Audio Artifact Timeline",
    )
    return fig

# --- Artifact timestamps ---
def get_artifact_timestamps(preds, chunk_duration):
    timestamps = []
    for i, pred in enumerate(preds):
        if pred != 0:
            start = i * chunk_duration
            end = (i + 1) * chunk_duration
            timestamps.append((start, end, f"{start:.2f}s - {end:.2f}s", class_label_map[pred]))
    return timestamps

# --- Live waveform ---
def live_waveform_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', line=dict(color='gray')))
    fig.update_layout(height=200, title="Waveform", xaxis_title="Samples", yaxis_title="Amplitude")
    st.plotly_chart(fig, use_container_width=True)

# --- Audio buffer (Uses the sample rate passed to it) ---
@st.cache_data
def get_audio_buffer(y, samplerate):
    audio_buffer = io.BytesIO()
    # Writes the audio data 'y' using the specified 'samplerate'
    sf.write(audio_buffer, y, samplerate, format="WAV") 
    audio_buffer.seek(0)
    return audio_buffer.read()

# --- Two-column layout ---
col1, col2 = st.columns([1,2])

# Initialize session state for data persistence across Streamlit reruns
if 'y_for_plot' not in st.session_state:
    st.session_state['y_for_plot'] = None
if 'sr_for_plot' not in st.session_state:
    st.session_state['sr_for_plot'] = None
if 'preds' not in st.session_state:
    st.session_state['preds'] = None
if 'chunk_duration' not in st.session_state:
    st.session_state['chunk_duration'] = None


with col1:
    st.header("Controls")
    
    # Model loading logic (unchanged)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "checkpoints")
    try:
        model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
    except FileNotFoundError:
        st.error(f"Error: Checkpoints directory not found at {weights_dir}")
        model_files = [] # Prevent error if directory doesn't exist

    selected_model = st.selectbox("Select model weights", model_files)
    weights_path = os.path.join(weights_dir, selected_model)
    model = load_model_cached(weights_path)

    input_type = st.radio("Input type", ["Upload Audio File", "Live Line-in"])

    # --- UPLOAD AUDIO FILE LOGIC (FIXED) ---
    if input_type == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload audio", type=["wav","mp3"])
        if uploaded_file is not None:
            
            # 1. Load original audio and sample rate for playback
            y_original, sr_original = librosa.load(uploaded_file, sr=None)
            
            # 2. Create resampled version for model inference
            if sr_original != params.SAMPLE_RATE:
                # Resample using torchaudio utility for consistency with model processing
                y_tensor = torch.tensor(y_original).unsqueeze(0)
                y_resampled = TAF.resample(y_tensor, orig_freq=sr_original, new_freq=params.SAMPLE_RATE).squeeze(0).numpy()
            else:
                y_resampled = y_original
            
            # 3. Process the resampled audio with the model
            progress_bar = st.progress(0)
            preds, chunk_duration = process_audio_chunked(
                model, 
                y_resampled, # Process the resampled data
                params.SAMPLE_RATE, # Use the resampled rate
                update_progress=progress_bar.progress
            )
            
            # 4. Store *original* data and rate for correct playback/plotting
            st.session_state['y_for_plot'] = y_original
            st.session_state['sr_for_plot'] = sr_original
            st.session_state['preds'] = preds
            st.session_state['chunk_duration'] = chunk_duration
            
            st.subheader("Audio")
            st.audio(get_audio_buffer(y_original, sr_original), format="audio/wav")

    # --- LIVE LINE-IN LOGIC (UNCHANGED) ---
    else:
        st.write("Press 'Start Recording' to capture 10s of audio.")
        if st.button("Start Recording"):
            st.info("Recording in 3...2...1...")
            time.sleep(1)
            q = queue.Queue()
            recorded = []
            
            # Recording is already done at params.SAMPLE_RATE
            def callback(indata, frames, time, status):
                q.put(indata.copy())
            
            with sd.InputStream(callback=callback, channels=1, samplerate=params.SAMPLE_RATE):
                start_time = time.time()
                while time.time()-start_time<10:
                    if not q.empty():
                        data = q.get()
                        recorded.extend(data.flatten())
                        live_waveform_plot(np.array(recorded[-params.SAMPLE_RATE*2:]))
                        
            st.success("Recording complete ✅")
            y_recorded = np.array(recorded)
            
            # Process the recorded audio
            progress_bar = st.progress(0)
            preds, chunk_duration = process_audio_chunked(
                model, 
                y_recorded, 
                params.SAMPLE_RATE, # Already at the correct rate
                update_progress=progress_bar.progress
            )
            
            # Store recorded data for correct plotting/playback
            st.session_state['y_for_plot'] = y_recorded
            st.session_state['sr_for_plot'] = params.SAMPLE_RATE
            st.session_state['preds'] = preds
            st.session_state['chunk_duration'] = chunk_duration

# --- OUTPUTS COLUMN ---
with col2:
    st.header("Outputs")
    
    preds = st.session_state['preds']
    chunk_duration = st.session_state['chunk_duration']
    y_plot = st.session_state['y_for_plot']
    
    if preds is not None:
        fig = plot_timeline(preds, chunk_duration)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Waveform Preview")
        if y_plot is not None:
            # Plot uses the stored audio data
            live_waveform_plot(y_plot)
        
        artifact_ts = get_artifact_timestamps(preds, chunk_duration)
        st.subheader("Artifacts Detected")
        
        if artifact_ts:
            df = pd.DataFrame({
                "Start Time": [round(ts[0], 2) for ts in artifact_ts],
                "End Time": [round(ts[1], 2) for ts in artifact_ts],
                "Time Range": [ts[2] for ts in artifact_ts],
                "Label": [ts[3] for ts in artifact_ts]
            })
            df_style = df.style.set_properties(**{'text-align': 'left'})
            st.dataframe(df_style)
        else:
            st.write("No artifacts detected.")
    else:
        st.info("Please upload an audio file or start a live recording to see results.")