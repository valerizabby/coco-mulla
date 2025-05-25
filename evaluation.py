import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

DURATION_SEC = 5  # duration of music tracks to crop
SR = 32000        # disc frequency

def compute_metrics(gt_audio, gen_audio, sr):
    def chroma(x): return librosa.feature.chroma_cqt(y=x, sr=sr).mean(axis=1)
    def energy(x): return librosa.feature.rms(y=x).squeeze()

    gt_chroma = chroma(gt_audio)
    gen_chroma = chroma(gen_audio)

    gt_energy = energy(gt_audio)
    gen_energy = energy(gen_audio)

    pitch_corr = pearsonr(gt_chroma, gen_chroma)[0]
    energy_corr = pearsonr(gt_energy, gen_energy)[0]

    energy_ratio = np.mean(gen_energy) / (np.mean(gt_energy) + 1e-6)

    return pitch_corr, energy_corr, energy_ratio

def load_and_crop(path, duration_sec=DURATION_SEC, sr=SR):
    audio, _ = librosa.load(path, sr=sr)
    max_len = int(duration_sec * sr)
    return audio[:max_len]

def process_track(track_path):
    result = {"track": os.path.basename(track_path)}
    gt_path = os.path.join(track_path, "audio", "audio.wav")
    if not os.path.exists(gt_path):
        print(f"No ground truth found for {track_path}")
        return result

    gt_audio = load_and_crop(gt_path)

    gen_paths = {
        "musicgen": os.path.join(track_path, "musicgen_output.wav"),
        "chord-only": os.path.join(track_path, "cocomulla_output", "chord-only.wav.wav"),
        "chord-drums": os.path.join(track_path, "cocomulla_output", "chord-drums.wav.wav"),
        "chord-midi": os.path.join(track_path, "cocomulla_output", "chord-midi.wav.wav"),
        "chord-drums-midi": os.path.join(track_path, "cocomulla_output", "chord-drums-midi.wav.wav"),
    }

    for key, path in gen_paths.items():
        if not os.path.exists(path):
            print(f"⚠File missing: {path}")
            result[f"{key}_pitch_corr"] = np.nan
            result[f"{key}_energy_corr"] = np.nan
            result[f"{key}_energy_ratio"] = np.nan
            continue

        gen_audio = load_and_crop(path)
        pitch_corr, energy_corr, energy_ratio = compute_metrics(gt_audio, gen_audio, SR)
        result[f"{key}_pitch_corr"] = pitch_corr
        result[f"{key}_energy_corr"] = energy_corr
        result[f"{key}_energy_ratio"] = energy_ratio

    return result

def evaluate_all(base_dir):
    rows = []
    for track_folder in tqdm(sorted(os.listdir(base_dir))):
        track_path = os.path.join(base_dir, track_folder)
        if not os.path.isdir(track_path):
            continue
        try:
            row = process_track(track_path)
            rows.append(row)
        except Exception as e:
            print(f"Failed to process {track_folder}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv("generation_metrics.csv", index=False)

    print(df.drop(columns=["track"]).mean(numeric_only=True))

if __name__ == "__main__":
    evaluate_all("/Users/valerizab/Desktop/aml/coco-mulla-repo/testset")