import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

SOURCE_FOLDER = "Maharashtra"
DEST_FOLDER = "Maharashtra_Spectrograms"
IMG_SIZE = (128, 128)
CLIP_DURATION = 5    
SAMPLE_RATE = 22050   
ADD_NOISE = True      

F_MIN = 500
F_MAX = 10000 

print(f"reading from: {SOURCE_FOLDER}")
print(f"saving to: {DEST_FOLDER}")

if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)

species_list = [d for d in os.listdir(SOURCE_FOLDER) if os.path.isdir(os.path.join(SOURCE_FOLDER, d))]
species_list.sort() 

print("\nIMPORTANT: Class Mapping:")
for index, species in enumerate(species_list):
    print(f"{index} : {species}")

for species in species_list:
    input_path = os.path.join(SOURCE_FOLDER, species)
    output_path = os.path.join(DEST_FOLDER, species)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    wav_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]
    
    count = 0
    for wav_file in wav_files:
        try:
            file_path = os.path.join(input_path, wav_file)
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            chunk_samples = CLIP_DURATION * sr
            total_duration = librosa.get_duration(y=y, sr=sr)
            
            if total_duration < 1:
                continue

            for i in range(0, len(y), chunk_samples):
                slice_audio = y[i : i + chunk_samples
                                ]
                if len(slice_audio) < chunk_samples:
                    padding = chunk_samples - len(slice_audio)
                    slice_audio = np.pad(slice_audio, (0, padding), 'constant')

                if ADD_NOISE:
                    noise = np.random.normal(0, 0.005, len(slice_audio))
                    slice_audio = slice_audio + noise

                S = librosa.feature.melspectrogram(
                    y=slice_audio, 
                    sr=sr, 
                    n_fft=2048, 
                    hop_length=512, 
                    n_mels=128, 
                    fmin=F_MIN, 
                    fmax=F_MAX
                )
                
                S_dB = librosa.power_to_db(S, ref=np.max)

                img_data = np.flip(S_dB, axis=0) 
                
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                img_data = (img_data * 255).astype(np.uint8)
                
                img = Image.fromarray(img_data)
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                
                fname = f"{species}_{wav_file[:-4]}_{i}.png"
                img.save(os.path.join(output_path, fname))
                count += 1
                
        except Exception as e:
            print(f"error processing {wav_file}: {e}")

    print(f"DONE: generated {count} spectrograms for {species}.")