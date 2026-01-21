import os
import numpy as np
import librosa
import tensorflow as tf
from PIL import Image
from tqdm import tqdm 

MODEL_PATH = "maharashtra_birds_model.h5"
AUDIO_FILE = "./Maharashtra/Malabar_Whistling_Thrush_folder/Malabar_Whistling_Thrush_14.wav" # CHANGE THIS to your audio file name

# keep in alphabetical order
CLASSES = [
    "Coppersmith Barbet", 
    "Common Myna", 
    "House Crow", 
    "Indian Peafowl", 
    "Malabar Grey Hornbill", 
    "Malabar Whistling Thrush"
]

NATIVE_BIRDS = ["Indian Peafowl", "Malabar Whistling Thrush", "Malabar Grey Hornbill", "Coppersmith Barbet"]
INVASIVE_BIRDS = ["House Crow", "Common Myna"]

CONFIDENCE_THRESHOLD = 0.75 

def get_spectrogram(audio_slice, sr):
    S = librosa.feature.melspectrogram(
        y=audio_slice, 
        sr=sr, 
        n_fft=2048, 
        hop_length=512, 
        n_mels=128, 
        fmin=500, # wind filter
        fmax=10000
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    img = np.flip(S_dB, axis=0)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img).convert('RGB')
    img_pil = img_pil.resize((128, 128))
    
    return np.array(img_pil)

def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"- processing audio: {AUDIO_FILE} -")
    
    y, sr = librosa.load(AUDIO_FILE, sr=22050)
    total_duration = librosa.get_duration(y=y, sr=sr)
    print(f"audio duration: {total_duration:.2f} seconds")

    chunk_samples = 5 * sr
    predictions = []
    
    total_chunks = int(len(y) / chunk_samples)
    for i in tqdm(range(0, len(y), chunk_samples), total=total_chunks):
        slice_audio = y[i : i + chunk_samples]

        if len(slice_audio) < chunk_samples:
            continue
        spec_img = get_spectrogram(slice_audio, sr)
        
        input_data = np.expand_dims(spec_img, axis=0)
        
        pred_probs = model.predict(input_data, verbose=0)[0]
        
        best_index = np.argmax(pred_probs)
        confidence = pred_probs[best_index]
        
        if confidence >= CONFIDENCE_THRESHOLD:
            bird_name = CLASSES[best_index]
            predictions.append((bird_name, confidence))
    
    if len(predictions) == 0:
        print("no birds detected (or sounds were too faint).")
        return
    
    unique_birds = set([p[0] for p in predictions])
    print(f"\nspecies Detected: {len(unique_birds)}")
    
    score = 50 
    
    for bird in unique_birds:
        scores = [p[1] for p in predictions if p[0] == bird]
        avg_conf = sum(scores) / len(scores)
        print(f" > {bird} (confidence: {avg_conf*100:.1f}%)")

        if bird in NATIVE_BIRDS:
            print(f"   [+] Native Species (+15 pts)")
            score += 15
        elif bird in INVASIVE_BIRDS:
            print(f"   [-] Invasive Species (-20 pts)")
            score -= 20
        else:
            print(f"   [?] Neutral Species (0 pts)")

    score = max(0, min(100, score))
    
    print(f"FINAL FOREST HEALTH SCORE: {score}/100")
    
    if score >= 80:
        print("Condition: EXCELLENT / PRISTINE")
    elif score >= 50:
        print("Condition: MODERATE / STABLE")
    else:
        print("Condition: CRITICAL / DEGRADED")

if __name__ == "__main__":
    main()