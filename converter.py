import os
import subprocess

raw_dir = "./bird_raw_data"
wav_dir = "./bird_raw_data_wav"

if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

folders = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

for fol in folders:
    in_p = os.path.join(raw_dir, fol)
    out_p = os.path.join(wav_dir, fol)
    
    print(f"\n-checking folder: {fol}-")
    os.makedirs(out_p, exist_ok=True)
    
    files = [f for f in os.listdir(in_p) if f.endswith(".mp3")]
    
    for i, f_name in enumerate(files):
        mp3_file = os.path.join(in_p, f_name)
        wav_name = f_name.replace(".mp3", ".wav")
        wav_file = os.path.join(out_p, wav_name)

        if os.path.exists(wav_file):
            continue

        print(f"[{i+1}/{len(files)}] fixing & converting: {f_name}")

        cmd = [
            'ffmpeg', '-y', '-i', mp3_file, 
            '-acodec', 'pcm_s16le', 
            '-ar', '44100', 
            '-map_metadata', '-1', 
            wav_file
        ]

        # debugging
        # print(cmd)
        # print(wav_name)
        # exit(1) 

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"failed to convert {f_name}, might be totally corrupt")

print("\nDONE")