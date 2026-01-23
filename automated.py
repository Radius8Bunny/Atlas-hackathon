import requests
import os
import time

class XC:
    def __init__(self, api_key, bp="./bird_raw_data"):
        self.api_key = api_key
        self.bp = bp
        self.endpoint = "https://xeno-canto.org/api/3/recordings"
        
        if not os.path.exists(self.bp):
            os.makedirs(self.bp)

    def download_bird(self, bird_name, limit=30):
        c_name = bird_name.replace(" ", "_")
        folname = f"{c_name}_folder"
        tdir = os.path.join(self.bp, folname)
        
        print(f"\n-processing: {bird_name}-")
        os.makedirs(tdir, exist_ok=True)
        query_string = f'en:"{bird_name}" q:A'
        
        params = {
            "query": query_string,
            "key": self.api_key,
            "page": 1
        }

        try:
            resp = requests.get(self.endpoint, params=params)
            if resp.status_code != 200:
                print(f"error {resp.status_code}: {resp.text}")
                return

            data = resp.json()
            recordings = data.get('recordings', [])
            
            print(f"found {data.get('numRecordings')} A qual")

            d_count = 0
            for i, rec in enumerate(recordings):
                if d_count >= limit:
                    break
                
                file_url = rec.get('file')
                if not file_url: continue
                if file_url.startswith("//"): file_url = "https:" + file_url
                file_name = f"{c_name}_{d_count}.mp3"
                save_path = os.path.join(tdir, file_name)
                if os.path.exists(save_path):
                    print(f"- skipping {file_name} (already exists)")
                    d_count += 1
                    continue

                print(f"[{d_count+1}/{limit}] saving {file_name}")
                
                audio_r = requests.get(file_url)
                if audio_r.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(audio_r.content)
                    d_count += 1
                    time.sleep(1)
                else:
                    print(f"- failed to get file for {file_name}")

        except Exception as e:
            print(f"- error with {bird_name}: {e}")

API_KEY = "b1bd63fa57bf57fa32fec5fc31e4da44b4f66dd4"

birds_to_get = [
    # "Bay-backed Shrike",
    # "Brahminy Kite",
    # "Chestnut-bellied Sandgrouse",
    # "Common Woodshrike",
    # "Demoiselle Crane",
    # "Dusky Eagle-Owl",
    # "Forest Wagtail",
    # "Great Grey Shrike", # f
    # "Great Hornbill",
    # "Green-crowned Warbler",
    # "Himalayan Rubythroat",
    # "Indian Blue Robin",
    # "Indian Paradise Flycatcher", # f
    # "Indian stone-curlew", # f
    # "Kentish Plover",
    # "Little Bunting",
    # "Orange-headed thrush",  # f
    # "Pied Bush Chat", # f
    # "Singing bush lark", # f
    # "Small Minivet",
    # "Streaked Weaver",
    # "Thick-billed Flowerpecker",
    # "Tytler's Leaf Warbler",
    # "Velvet-fronted Nuthatch",
    # "Yellow-crowned woodpecker" # f
    "Asian Koel",
    "Black Kite",
    "Common Rosefinch",
    "Greater Coucal",
    "Hypocolius",
    "Indian Golden Oriole",
    "Indian Pied Starling",
    "Indian Pond-Heron",
    "Rosy Starling",
    "Rufous Treepie",
    "White-rumped Munia",
    "White-rumped Vulture",
    "Yellow-legged Buttonquail"
]
scraper = XC(api_key=API_KEY)

for bird in birds_to_get:
    scraper.download_bird(bird, limit=30)

print("\nDONE")