import os
import librosa
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_FILE = "final_brain.keras" 
# verify class names indexing always before running
CLASSES = [
    "Common Myna",              
    "Coppersmith Barbet",        
    "House Crow",              
    "Indian Peafowl",         
    "Malabar Grey Hornbill",    
    "Malabar Whistling Thrush"  
]
NATIVES = ["Indian Peafowl", "Malabar Whistling Thrush", "Malabar Grey Hornbill", "Coppersmith Barbet"]
INVASIVES = ["House Crow", "Common Myna"]

CONFIDENCE_CUTOFF = 0.88 

def get_spectrogram(chunk, sr):
    s = librosa.feature.melspectrogram(
        y=chunk, sr=sr, n_fft=2048, hop_length=512, 
        n_mels=128, fmin=500, fmax=10000
    )
    db = librosa.power_to_db(s, ref=np.max)
    
    img = (db - db.min()) / (db.max() - db.min())
    img = (img * 255).astype(np.uint8)
    img = np.flip(img, axis=0)
    
    return Image.fromarray(img).convert("RGB").resize((224, 224))

def calculate_ecology_metrics(hits):
    if not hits:
        return 0, [], 0, 0.0

    native_calls = [h for h in hits if h in NATIVES]
    invasive_calls = [h for h in hits if h in INVASIVES]
    
    # N = unique native species
    found_natives = sorted(list(set(native_calls)))
    n_score = len(found_natives)
    
    # D = shannon diversity index
    d_score = 0.0
    if n_score > 1:
        counts = [native_calls.count(bird) for bird in found_natives]
        total = len(native_calls)
        probs = [c / total for c in counts]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        d_score = entropy / np.log(n_score)
    elif n_score == 1:
        d_score = 1.0

    # I = invasive impact 
    i_score = len(invasive_calls)
    
    # H = (N * D) - I formula 
    raw_h = (n_score * d_score * 12) - (i_score * 0.75)
    final_h = int(max(0, min(100, (raw_h * 1.5) + 45)))
    
    return final_h, found_natives, i_score, d_score

def run_analysis(audio_path):
    if not os.path.exists(audio_path):
        print(f"file error: {audio_path} not found.")
        return

    print("loading the model") # remove in final product, used for debugging
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 5% overlap
    win_len = 5 * sr
    hop_len = int(2.5 * sr)
    detections = []

    for start in range(0, len(y) - win_len, hop_len):
        segment = y[start : start + win_len]
        spec_img = get_spectrogram(segment, sr)
        
        x = np.expand_dims(np.array(spec_img), axis=0)
        preds = model.predict(x, verbose=0)[0]
        
        idx = np.argmax(preds)
        if preds[idx] >= CONFIDENCE_CUTOFF:
            detections.append(CLASSES[idx])


    h_score, natives, i_count, diversity = calculate_ecology_metrics(detections)

    print(f"HEALTH SCORE:   {h_score}/100")
    print(f"DIVERSITY (D):  {diversity:.2f}")
    print(f"INVASIVE LOAD:  {i_count} detections")
    print("\n")
    print(f"NATIVE SPECIES DETECTED:")
    if natives:
        for n in natives: print(f"- {n}")
    else:
        print("- (None detected)")


def main(input_track):
    # run_analysis("Maharashtra/Malabar_Whistling_Thrush_folder/Malabar_Whistling_Thrush_7.wav")
    run_analysis(input_track)