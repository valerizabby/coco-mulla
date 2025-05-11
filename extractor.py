import pandas as pd
import os
import json

# .parquet-file (LMD metadata)
df = pd.read_parquet("LMD/train-00000-of-00001.parquet")

# test set directory
root_dir = "testset"

for track_folder in os.listdir(root_dir):
    track_path = os.path.join(root_dir, track_folder)
    if not os.path.isdir(track_path):
        continue

    audio_dir = os.path.join(track_path, "audio")
    if not os.path.exists(audio_dir):
        print(f"No audio directory in {track_folder}")
        continue

    mp3_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]
    if len(mp3_files) != 1:
        print(f"Only one .mp3 was expected in {audio_dir}, found: {len(mp3_files)}")
        continue

    track_id = mp3_files[0].replace(".mp3", "")

    entry = df[df['track_id'] == track_id]
    if entry.empty:
        print(f"No metadata for {track_id}")
        continue

    title = entry["title"].values[0]
    artist = entry["artist"].values[0]
    chords = entry["chords"].values[0]
    symbols = chords["symbol"]
    timestamps = chords["timestamp"]

    meta_path = os.path.join(track_path, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "track_id": track_id,
            "title": title,
            "artist": artist
        }, f, indent=2)

    lab_path = os.path.join(track_path, "chords.lab")
    with open(lab_path, "w") as f:
        for symbol, ts in zip(symbols, timestamps):
            f.write(f"{ts:.3f}\t{symbol}\n")

    print(f"Saved: {track_id}")