import os
import librosa
import numpy as np
import tensorflow as tf
from PIL import Image

# config
MODEL_PATH = "final_brain_v3.keras"
CUTOFF = 0.78
# class labels 
LABELS = ["Asian_Koel", "Bay-backed_Shrike", "Black_Kite", "Brahminy_Kite", "Chestnut-bellied_Sandgrouse", "Common_Myna", "Common_Rosefinch", "Common_Woodshrike", "Coppersmith_Barbet", "Demoiselle_Crane", "Dusky_Eagle-Owl", "Forest_Wagtail", "Great_Grey_Shrike", "Great_Hornbill", "Greater_Coucal", "Green-crowned_Warbler", "Himalayan_Rubythroat", "House_Crow", "Hypocolius", "Indian_Blue_Robin", "Indian_Golden_Oriole", "Indian_Paradise_Flycatcher", "Indian_Peafowl", "Indian_Pied_Starling", "Indian_Pond-Heron", "Indian_stone-curlew", "Kentish_Plover", "Little_Bunting", "Malabar_Grey_Hornbill", "Malabar_Whistling_Thrush", "Pied_Bush_Chat", "Rufous_Treepie", "Singing_bush_lark", "Small_Minivet", "Streaked_Weaver", "Thick-billed_Flowerpecker", "Tytlers_Leaf_Warbler", "Velvet-fronted_Nuthatch", "White-rumped_Munia", "White-rumped_Vulture", "Yellow-legged_Buttonquail"]

NATIVE_BIRDS = {"Indian_Peafowl", "Malabar_Whistling_Thrush", "Malabar_Grey_Hornbill", "Coppersmith_Barbet", "Bay-backed_Shrike", "Brahminy_Kite", "Chestnut-bellied_Sandgrouse", "Common_Woodshrike", "Demoiselle_Crane", "Dusky_Eagle-Owl", "Forest_Wagtail", "Great_Grey_Shrike", "Great_Hornbill", "Green-crowned_Warbler", "Himalayan_Rubythroat", "Indian_Blue_Robin", "Indian_Paradise_Flycatcher", "Indian_stone-curlew", "Kentish_Plover", "Little_Bunting", "Pied_Bush_Chat", "Singing_bush_lark", "Small_Minivet", "Streaked_Weaver", "Thick-billed_Flowerpecker", "Tytlers_Leaf_Warbler", "Velvet-fronted_Nuthatch", "White-rumped_Vulture", "Yellow-legged_Buttonquail"}

INVASIVE_BIRDS = {"House_Crow", "Common_Myna", "Asian_Koel", "Black_Kite", "Common_Rosefinch", "Greater_Coucal", "Hypocolius", "Indian_Golden_Oriole", "Indian_Pied_Starling", "Indian_Pond-Heron", "Rufous_Treepie", "White-rumped_Munia"}

def get_spec(y, sr):
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=500, fmax=10000)
    db = librosa.power_to_db(s, ref=np.max)
    norm = (db - db.min()) / (db.max() - db.min())
    img = Image.fromarray((np.flip(norm, axis=0) * 255).astype(np.uint8))
    return img.convert("RGB").resize((224, 224))

def analyze_audio(f):
    if not os.path.exists(f): return print("no file")
    
    print(f"analyzing: {f}")
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    sig, rate = librosa.load(f, sr=22050)
    
    # windowing
    w_len = 5 * rate
    step = int(2.5 * rate)
    hits = [] 

    for i in range(0, len(sig) - w_len, step):
        patch = sig[i : i + w_len]
        img = get_spec(patch, rate)
        
        arr = np.expand_dims(np.array(img), 0)
        p = m.predict(arr, verbose=0)[0]
        best = np.argmax(p)
        
        if p[best] >= CUTOFF:
            hits.append((LABELS[best], p[best]))

    if not hits: return print("nothing detected.")

    # eco stats
    names = [h[0] for h in hits]
    n_hits = [n for n in names if n in NATIVE_BIRDS]
    i_hits = [n for n in names if n in INVASIVE_BIRDS]
    unq_n = sorted(list(set(n_hits)))
    
    # diversity index
    div = 0.0
    if len(unq_n) > 1:
        pk = [n_hits.count(b) / len(n_hits) for b in unq_n]
        div = -sum(p * np.log(p) for p in pk if p > 0) / np.log(len(unq_n))
    elif len(unq_n) == 1:
        div = 1.0

    # final scoring logic
    score = int(max(0, min(100, 30 + (len(unq_n) * 10) + (div * 15) - (len(i_hits) * 2))))

    print(f"\n--- ECOLOGY REPORT ---")
    print(f"HEALTH: {score}/100 | DIVERSITY: {div:.2f} | INVASIVES: {len(i_hits)}")
    print(f"{'BIRD':<25} | {'FREQ %':<8} | {'CONF'}")
    
    for b in set(names):
        pct = (names.count(b) / len(names)) * 100
        cf = np.mean([h[1] for h in hits if h[0] == b]) * 100
        print(f"{b:<25} | {pct:>7.1f}% | {cf:>5.1f}%")

    print("\nNATIVES:", ", ".join(unq_n) if unq_n else "None")


def main(inp):
    analyze_audio(inp)