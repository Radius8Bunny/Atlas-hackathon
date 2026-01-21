import os
import shutil

def parse_state_file(filepath):
    data = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            exec(content, {}, data)
    except Exception as e:
        print(f"- error reading text file: {e} -")
    return data

def process_list(bird_list, target_path, raw_wav_dir):
    for bird in bird_list:
        clean_b = bird.replace(" ", "_")
        fol_name = f"{clean_b}_folder"
        
        src = os.path.join(raw_wav_dir, fol_name)
        dst = os.path.join(target_path, fol_name)

        if os.path.exists(dst):
            if os.path.exists(src):
                print(f"-> merging new files for {bird} into existing folder")
                for f in os.listdir(src):
                    s_file = os.path.join(src, f)
                    d_file = os.path.join(dst, f)
                    if not os.path.exists(d_file):
                        shutil.move(s_file, d_file)
                os.rmdir(src)
            else:
                print(f"- {bird} already sorted and no new data found")

        elif os.path.exists(src):
            print(f"-> moving {bird} to {target_path}")
            shutil.move(src, dst)
        
        else:
            print(f"! {bird} not found in raw data")

def sort_birds(config_file, raw_wav_dir="./bird_raw_data_wav"):
    cfg = parse_state_file(config_file)
    if not cfg: return

    loc = cfg.get('location', 'Unknown_State')
    native = cfg.get('native_birds', [])
    invasive = cfg.get('invasive_birds', [])

    root_dir = os.path.join("./", loc)
    subfolders = {
        "native": os.path.join(root_dir, "native"),
        "invasive": os.path.join(root_dir, "invasive")
    }

    for path in subfolders.values():
        os.makedirs(path, exist_ok=True)

    print(f"\n- sorting for {loc} -")
    process_list(native, subfolders["native"], raw_wav_dir)
    process_list(invasive, subfolders["invasive"], raw_wav_dir)

sort_birds("maharashtra.txt")
print("\nDONE")