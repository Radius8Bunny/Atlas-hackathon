import os
import numpy as np
import librosa
from pathlib import Path
from PIL import Image

def build_spectrogram(audio_path, output_path, img_size=(224, 224)):
    y, sr = librosa.load(audio_path, sr=22050)
    
    duration = librosa.get_duration(y=y, sr=sr)
    chunk_size = 5 * sr
    
    for start in range(0, len(y), chunk_size):
        end = start + chunk_size
        if end > len(y): break
        
        slice_y = y[start:end]
        # hi-res fft
        S = librosa.feature.melspectrogram(
            y=slice_y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=400, fmax=10000
        )
        db = librosa.power_to_db(S, ref=np.max)
        
        img = (db - db.min()) / (db.max() - db.min())
        img = (img * 255).astype(np.uint8)
        img = np.flip(img, axis=0) 
        
        res = Image.fromarray(img).resize(img_size, Image.LANCZOS)
        res.convert("RGB").save(f"{output_path}_{start}.png")

def run_extraction(raw_dir, target_dir):
    p = Path(raw_dir)
    for folder in p.iterdir():
        if folder.is_dir():
            out = Path(target_dir) / folder.name
            out.mkdir(parents=True, exist_ok=True)
            print(f"Extracting: {folder.name}")
            for audio in folder.glob("*.wav"):
                build_spectrogram(audio, out / audio.stem)

if __name__ == "__main__":
    run_extraction("Maharashtra", "Maharashtra_Spectrograms")