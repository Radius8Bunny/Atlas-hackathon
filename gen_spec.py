import os
import numpy as np
import librosa
import glob
from PIL import Image

# simple spec maker
def make_spec(f, out, sz=(224, 224)):
    sig, rate = librosa.load(f, sr=22050)
    
    step = 5 * rate
    for i in range(0, len(sig), step):
        if i + step > len(sig): break
        
        chunk = sig[i:i+step]
        # mel stuff
        s = librosa.feature.melspectrogram(y=chunk, sr=rate, n_fft=2048, 
                                           hop_length=512, n_mels=128, fmin=400, fmax=10000)
        
        # log scale and norm
        d = librosa.power_to_db(s, ref=np.max)
        img = (d - d.min()) / (d.max() - d.min())
        img = (img * 255).astype(np.uint8)
        img = np.flip(img, axis=0) 

        # save it
        res = Image.fromarray(img).resize(sz, Image.LANCZOS)
        res.convert("RGB").save(f"{out}_{i}.png")

def do_all(src, dst):
    for d in os.listdir(src):
        d_path = os.path.join(src, d)
        if os.path.isdir(d_path):
            out_folder = os.path.join(dst, d)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
            print(f"working on {d}...")
            files = glob.glob(os.path.join(d_path, "*.wav"))
            for f in files:
                fname = os.path.splitext(os.path.basename(f))[0]
                save_path = os.path.join(out_folder, fname)
                make_spec(f, save_path)

if __name__ == "__main__":
    do_all("bird_raw_data_wav", "Maharashtra_Spectrograms")